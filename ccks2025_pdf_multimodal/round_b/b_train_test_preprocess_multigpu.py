#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多GPU并行数据预处理脚本
支持使用多张GPU加速图片向量生成
使用方法: python b_train_test_preprocess_multigpu.py --gpus 0,1,2,3
"""

import argparse
import fitz  # PyMuPDF
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
# 不要在主进程导入torch，让子进程独立导入
from warnings import filterwarnings
filterwarnings("ignore")


def pdf_to_jpg(args):
    """将单个PDF转换为JPG图片"""
    file_name, base_dir = args
    try:
        pdf_document = fitz.open(base_dir + '/documents/' + file_name)
        output_dir = base_dir + '/pdf_img/' + file_name.split('.')[0]
        os.makedirs(output_dir, exist_ok=True)

        for i in range(pdf_document.page_count):
            page = pdf_document.load_page(i)
            pix = page.get_pixmap(dpi=600)
            pix.save(output_dir + '/' + str(i+1) + '.jpg')

        pdf_document.close()
        return True
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False


def convert_pdfs_to_images(base_dir, num_workers=8):
    """多进程并行转换PDF为图片"""
    print(f"\n{'='*60}")
    print(f"步骤1: PDF转JPG (使用{num_workers}个进程)")
    print(f"{'='*60}")

    pdf_file_list = [x for x in os.listdir(base_dir + '/documents/') if 'pdf' in x]
    print(f"找到 {len(pdf_file_list)} 个PDF文件")

    # 使用多进程处理
    pool = mp.Pool(processes=num_workers)
    args_list = [(f, base_dir) for f in pdf_file_list]

    results = list(tqdm(
        pool.imap(pdf_to_jpg, args_list),
        total=len(pdf_file_list),
        desc="转换PDF"
    ))

    pool.close()
    pool.join()

    success_count = sum(results)
    print(f"✓ 成功转换 {success_count}/{len(pdf_file_list)} 个PDF文件\n")


def process_images_on_gpu(gpu_id, pdf_files, base_dir, output_queue):
    """在指定GPU上处理图片生成向量"""
    # 必须在导入任何torch相关模块之前设置环境变量
    import os
    import sys

    # 设置当前进程只能看到指定的GPU，并映射为设备0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MAX_PIXELS"] = '1229312'

    # 清除已经导入的torch缓存（如果有）
    if 'torch' in sys.modules:
        del sys.modules['torch']
    if 'transformers' in sys.modules:
        del sys.modules['transformers']

    # 现在导入GME模型
    from gme_inference import GmeQwen2VL
    import torch

    # 验证GPU设置
    print(f"[进程 GPU {gpu_id}] 可见GPU数量: {torch.cuda.device_count()}, 当前设备: {torch.cuda.current_device()}")

    # 加载模型到当前GPU（现在GPU {gpu_id}被映射为设备0）
    gme = GmeQwen2VL(
        model_name=base_dir.replace('/patent_b/train', '').replace('/patent_b/test', '') +
                   '/llm_model/iic/gme-Qwen2-VL-7B-Instruct',
        max_image_tokens=1280,
        device='cuda:0'  # 使用映射后的设备0
    )

    local_vectors = []
    local_page_nums = []
    local_file_names = []

    for pdf_file in tqdm(pdf_files, desc=f"GPU {gpu_id}", position=gpu_id):
        file_name = pdf_file.split('.')[0]
        img_dir = base_dir + '/pdf_img/' + file_name

        if not os.path.exists(img_dir):
            continue

        file_list = sorted([x for x in os.listdir(img_dir) if 'jpg' in x])

        for img_file in file_list:
            image_path = img_dir + '/' + img_file
            try:
                e_text = gme.get_image_embeddings(images=[image_path])
                local_vectors.append(e_text[0].to('cpu').numpy())

                page_num = int(img_file.split('.')[0])
                local_page_nums.append(page_num)
                local_file_names.append(file_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # 将结果放入队列
    output_queue.put({
        'gpu_id': gpu_id,
        'vectors': np.array(local_vectors),
        'page_nums': local_page_nums,
        'file_names': local_file_names
    })


def generate_image_vectors_multigpu(base_dir, gpu_ids):
    """使用多GPU并行生成图片向量"""
    print(f"\n{'='*60}")
    print(f"步骤2: 生成图片向量 (使用GPU: {gpu_ids})")
    print(f"{'='*60}")

    # 获取所有PDF文件列表
    pdf_file_list = [x for x in os.listdir(base_dir + '/pdf_img/')]
    print(f"找到 {len(pdf_file_list)} 个PDF目录")

    # 将PDF列表分配到各个GPU
    num_gpus = len(gpu_ids)
    chunk_size = len(pdf_file_list) // num_gpus
    pdf_chunks = []

    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(pdf_file_list)
        pdf_chunks.append(pdf_file_list[start_idx:end_idx])
        print(f"GPU {gpu_ids[i]}: 处理 {len(pdf_chunks[i])} 个PDF ({start_idx}-{end_idx})")

    # 创建进程和队列
    output_queue = mp.Queue()
    processes = []

    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=process_images_on_gpu,
            args=(gpu_id, pdf_chunks[i], base_dir, output_queue)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 收集结果
    print("\n收集所有GPU的结果...")
    all_results = []
    while not output_queue.empty():
        all_results.append(output_queue.get())

    # 按GPU ID排序
    all_results.sort(key=lambda x: x['gpu_id'])

    # 合并所有向量
    all_vectors = np.vstack([r['vectors'] for r in all_results])
    all_page_nums = []
    all_file_names = []

    for r in all_results:
        all_page_nums.extend(r['page_nums'])
        all_file_names.extend(r['file_names'])

    print(f"✓ 生成了 {len(all_vectors)} 个图片向量 (维度: {all_vectors.shape})\n")

    return all_vectors, all_page_nums, all_file_names


def generate_question_vectors(base_dir, jsonl_file, gpu_id=0):
    """生成问题向量（使用单GPU）"""
    print(f"\n{'='*60}")
    print(f"步骤3: 生成问题向量 (使用GPU {gpu_id})")
    print(f"{'='*60}")

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MAX_PIXELS"] = '1229312'

    from gme_inference import GmeQwen2VL

    # 加载模型
    gme = GmeQwen2VL(
        model_name=base_dir.replace('/patent_b/train', '').replace('/patent_b/test', '') +
                   '/llm_model/iic/gme-Qwen2-VL-7B-Instruct',
        max_image_tokens=1280
    )

    # 读取问题
    df_question = pd.read_json(jsonl_file, lines=True)
    print(f"找到 {len(df_question)} 个问题")

    # 生成向量
    question_vectors = np.empty((len(df_question), 3584))

    for i in tqdm(range(len(df_question)), desc="处理问题"):
        question = df_question.loc[i, 'question']
        query_vec = gme.get_text_embeddings(texts=[question])
        question_vectors[i] = query_vec[0].to('cpu').numpy()

    print(f"✓ 生成了 {len(question_vectors)} 个问题向量\n")

    return question_vectors


def process_dataset(base_dir, dataset_name, gpu_ids, num_workers=8):
    """处理单个数据集（train或test）"""
    print(f"\n{'#'*60}")
    print(f"# 处理 {dataset_name.upper()} 数据集")
    print(f"# 路径: {base_dir}")
    print(f"{'#'*60}")

    # 步骤1: PDF转JPG（多进程）
    convert_pdfs_to_images(base_dir, num_workers)

    # 步骤2: 生成图片向量（多GPU）
    img_vectors, page_nums, file_names = generate_image_vectors_multigpu(base_dir, gpu_ids)

    # 保存图片向量和映射关系
    img_page_num_mapping = pd.DataFrame({
        'index': range(len(page_nums)),
        'page_num': page_nums,
        'file_name': file_names
    })

    output_prefix = f'{dataset_name}_b'
    np.save(f'{output_prefix}_pdf_img_vectors.npy', img_vectors)
    img_page_num_mapping.to_csv(f'{output_prefix}_pdf_img_page_num_mapping.csv', index=False)

    print(f"✓ 已保存: {output_prefix}_pdf_img_vectors.npy")
    print(f"✓ 已保存: {output_prefix}_pdf_img_page_num_mapping.csv")

    # 步骤3: 生成问题向量（单GPU）
    jsonl_file = base_dir + f'/{dataset_name}.jsonl'
    if os.path.exists(jsonl_file):
        question_vectors = generate_question_vectors(base_dir, jsonl_file, gpu_ids[0])

        # 保存问题向量
        np.save(f'all_{output_prefix}_question_vectors.npy', question_vectors)
        print(f"✓ 已保存: all_{output_prefix}_question_vectors.npy")
    else:
        print(f"⚠ 未找到 {jsonl_file}，跳过问题向量生成")


def main():
    parser = argparse.ArgumentParser(description='多GPU并行数据预处理')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='使用的GPU ID，逗号分隔，例如: 0,1,2,3')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='PDF转JPG的进程数')
    parser.add_argument('--train-dir', type=str,
                        default='/usr/yuque/guo/pdf_processer/patent_b/train',
                        help='训练集目录')
    parser.add_argument('--test-dir', type=str,
                        default='/usr/yuque/guo/pdf_processer/patent_b/test',
                        help='测试集目录')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['train', 'test', 'both'],
                        help='处理哪个数据集')

    args = parser.parse_args()

    # 解析GPU ID
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]

    print(f"\n{'='*60}")
    print(f"多GPU并行数据预处理")
    print(f"{'='*60}")
    print(f"使用GPU: {gpu_ids}")
    print(f"PDF转JPG进程数: {args.num_workers}")
    print(f"处理数据集: {args.dataset}")

    # 处理训练集
    if args.dataset in ['train', 'both']:
        os.chdir('/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b')
        process_dataset(args.train_dir, 'train', gpu_ids, args.num_workers)

    # 处理测试集
    if args.dataset in ['test', 'both']:
        os.chdir('/usr/yuque/guo/pdf_processer/ccks2025_pdf_multimodal/round_b')
        process_dataset(args.test_dir, 'test', gpu_ids, args.num_workers)

    print(f"\n{'='*60}")
    print(f"✓ 全部完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()