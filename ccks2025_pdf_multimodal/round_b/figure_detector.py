# coding: utf-8
"""
使用 Qwen VL API 检测和分割 PDF 页面中的图片
"""

import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image
import re


@dataclass
class FigureBbox:
    """图片边界框信息"""
    figure_num: str  # 图片编号，如 "1", "2"
    x1: float
    y1: float
    x2: float
    y2: float
    width: float  # bbox 宽度
    height: float  # bbox 高度
    confidence: float = 0.9  # 检测置信度


@dataclass
class FigureMetadata:
    """图片元数据"""
    source_pdf: str  # PDF 文件名
    source_pdf_path: str  # PDF 完整路径
    page_num: int  # 页码（从1开始）
    figure_num: str  # 图片编号
    bbox: Dict[str, float]  # 边界框信息
    image_path: str  # 保存的图片路径
    image_filename: str  # 图片文件名
    image_size: Dict[str, int]  # 图片分辨率 {width, height}
    extracted_time: str  # 提取时间
    detection_prompt: str  # 用于检测的 prompt


class QwenVLFigureDetector:
    """使用 Qwen VL API 检测 PDF 页面中的图片"""

    def __init__(self, api_key: str, model: str = "qwen-vl-plus"):
        """
        初始化检测器

        Args:
            api_key: 阿里云百炼 API Key
            model: 使用的模型名称
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def detect_figures(self, image_path: str, page_height: int, page_width: int) -> List[FigureBbox]:
        """
        使用 Qwen VL 检测图片中的所有图表及其位置

        Args:
            image_path: 页面图片路径
            page_height: 页面高度（像素）
            page_width: 页面宽度（像素）

        Returns:
            检测到的图片列表，包含编号和边界框
        """
        # 编码图片
        image_base64 = self.encode_image_to_base64(image_path)

        # 构建提示词
        detection_prompt = """请仔细分析这个专利文档页面图片，找出所有标注编号的图（如图1、图2、图3等）。

对于每个找到的图片，请：
1. 识别其编号（例如：1、2、3等）
2. 估计其在页面中的位置，用归一化坐标表示（0-1之间）
3. 提供边界框坐标 [x1, y1, x2, y2]，其中：
   - x1, y1 是左上角坐标（相对于整个页面，范围0-1）
   - x2, y2 是右下角坐标（相对于整个页面，范围0-1）

请以JSON格式返回结果，格式如下：
{
    "figures": [
        {
            "figure_num": "1",
            "bbox": [x1, y1, x2, y2],
            "description": "图片简要描述（可选）"
        },
        {
            "figure_num": "2",
            "bbox": [x1, y1, x2, y2],
            "description": "图片简要描述（可选）"
        }
    ],
    "total_figures": 5
}

如果页面上没有找到编号的图片，请返回空的 figures 列表。"""

        try:
            # 调用 API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": detection_prompt
                            }
                        ]
                    }
                ],
                temperature=0.1,  # 降低温度以获得更稳定的结果
            )

            response_text = completion.choices[0].message.content
            print(f"\n[Qwen VL 原始响应]\n{response_text}\n")

            # 解析 JSON 响应
            # 尝试从响应中提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                print(f"警告：无法从响应中提取 JSON 格式数据")
                return []

            json_str = json_match.group(0)
            result = json.loads(json_str)

            # 转换为 FigureBbox 对象
            figures = []
            for fig in result.get("figures", []):
                bbox = fig.get("bbox", [0, 0, 1, 1])
                # 将归一化坐标转换为像素坐标
                x1, y1, x2, y2 = bbox

                figure_bbox = FigureBbox(
                    figure_num=str(fig.get("figure_num", "?")),
                    x1=int(x1 * page_width),
                    y1=int(y1 * page_height),
                    x2=int(x2 * page_width),
                    y2=int(y2 * page_height),
                    width=int((x2 - x1) * page_width),
                    height=int((y2 - y1) * page_height),
                    confidence=0.9
                )
                figures.append(figure_bbox)

            print(f"✓ 检测到 {len(figures)} 个图片")
            for fig in figures:
                print(f"  - 图{fig.figure_num}: bbox=({fig.x1}, {fig.y1}, {fig.x2}, {fig.y2})")

            return figures

        except json.JSONDecodeError as e:
            print(f"✗ JSON 解析失败: {e}")
            print(f"响应内容: {response_text}")
            return []
        except Exception as e:
            print(f"✗ API 调用失败: {e}")
            return []


class PDFFigureExtractor:
    """PDF 图片提取器"""

    def __init__(self, detector: QwenVLFigureDetector):
        """
        初始化提取器

        Args:
            detector: Qwen VL 图片检测器实例
        """
        self.detector = detector

    def pdf_to_images(self, pdf_path: str, output_dir: str, dpi: int = 600) -> List[Tuple[str, int]]:
        """
        将 PDF 转换为图片

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            dpi: 转换 DPI

        Returns:
            列表，每个元素是 (图片路径, 页码)
        """
        os.makedirs(output_dir, exist_ok=True)
        pdf_doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        page_images = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            # 转换为图片
            pix = page.get_pixmap(dpi=dpi, alpha=False)

            # 保存
            image_filename = f"{pdf_name}_page_{page_num + 1}_full.jpg"
            image_path = os.path.join(output_dir, image_filename)
            pix.save(image_path)

            page_images.append((image_path, page_num + 1))
            print(f"✓ 已保存页面图片: {image_filename}")

        pdf_doc.close()
        return page_images

    def extract_figures_from_page(
        self,
        page_image_path: str,
        page_num: int,
        pdf_name: str,
        output_dir: str
    ) -> Tuple[List[FigureMetadata], List[str]]:
        """
        从页面图片中提取所有图片

        Args:
            page_image_path: 页面图片路径
            page_num: 页码
            pdf_name: PDF 文件名（不含扩展名）
            output_dir: 输出目录

        Returns:
            元组: (元数据列表, 保存的图片路径列表)
        """
        # 加载页面图片
        page_img = cv2.imread(page_image_path)
        if page_img is None:
            print(f"✗ 无法读取页面图片: {page_image_path}")
            return [], []

        page_height, page_width = page_img.shape[:2]
        print(f"\n页面尺寸: {page_width}x{page_height}")

        # 使用 Qwen VL 检测图片
        print("正在使用 Qwen VL 检测图片...")
        figure_bboxes = self.detector.detect_figures(
            page_image_path,
            page_height,
            page_width
        )

        if not figure_bboxes:
            print(f"页面 {page_num} 未检测到任何图片")
            return [], []

        # 为每个检测到的图片创建目录
        figures_output_dir = os.path.join(output_dir, f"{pdf_name}_page_{page_num}")
        os.makedirs(figures_output_dir, exist_ok=True)

        metadata_list = []
        saved_images = []

        for bbox in figure_bboxes:
            # 裁剪图片
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            # 添加一些边距
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(page_width, x2 + margin)
            y2 = min(page_height, y2 + margin)

            cropped_img = page_img[y1:y2, x1:x2]

            # 保存图片
            image_filename = f"{pdf_name}_page_{page_num}_figure_{bbox.figure_num}.jpg"
            image_path = os.path.join(figures_output_dir, image_filename)
            cv2.imwrite(image_path, cropped_img)

            saved_images.append(image_path)

            # 记录元数据
            metadata = FigureMetadata(
                source_pdf=f"{pdf_name}.pdf",
                source_pdf_path=os.path.abspath(page_image_path).replace("_page_", "_").split("_full")[0] + ".pdf",
                page_num=page_num,
                figure_num=bbox.figure_num,
                bbox={
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                    "width": bbox.width,
                    "height": bbox.height
                },
                image_path=image_path,
                image_filename=image_filename,
                image_size={
                    "width": cropped_img.shape[1],
                    "height": cropped_img.shape[0]
                },
                extracted_time=datetime.now().isoformat(),
                detection_prompt="使用Qwen VL检测图片编号和位置"
            )
            metadata_list.append(metadata)

            print(f"✓ 已保存图片: {image_filename} (size: {cropped_img.shape[1]}x{cropped_img.shape[0]})")

        return metadata_list, saved_images

    def process_pdf(
        self,
        pdf_path: str,
        output_base_dir: str,
        dpi: int = 600
    ) -> Dict[str, Any]:
        """
        处理整个 PDF 文件

        Args:
            pdf_path: PDF 文件路径
            output_base_dir: 输出基础目录
            dpi: 转换 DPI

        Returns:
            处理结果字典
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(output_base_dir, pdf_name)

        print(f"\n{'='*60}")
        print(f"开始处理 PDF: {os.path.basename(pdf_path)}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}")

        # Step 1: PDF 转图片
        print("\n[Step 1] 将 PDF 转换为页面图片...")
        temp_dir = os.path.join(output_dir, "temp_full_pages")
        page_images = self.pdf_to_images(pdf_path, temp_dir, dpi)

        # Step 2: 为每一页检测和提取图片
        print("\n[Step 2] 检测和提取图片...")
        all_metadata = []
        total_figures = 0

        for page_image_path, page_num in page_images:
            print(f"\n--- 处理页面 {page_num} ---")
            metadata_list, saved_images = self.extract_figures_from_page(
                page_image_path,
                page_num,
                pdf_name,
                output_dir
            )
            all_metadata.extend(metadata_list)
            total_figures += len(saved_images)

        # Step 3: 保存元数据
        print("\n[Step 3] 保存元数据...")
        metadata_file = os.path.join(output_dir, "extraction_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(m) for m in all_metadata],
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"✓ 元数据已保存: {metadata_file}")

        # Step 4: 保存 CSV 映射表
        import pandas as pd
        csv_data = []
        for idx, metadata in enumerate(all_metadata):
            csv_data.append({
                "index": idx,
                "page_num": metadata.page_num,
                "figure_num": metadata.figure_num,
                "image_filename": metadata.image_filename,
                "image_path": metadata.image_path,
                "bbox_x1": metadata.bbox["x1"],
                "bbox_y1": metadata.bbox["y1"],
                "bbox_x2": metadata.bbox["x2"],
                "bbox_y2": metadata.bbox["y2"],
                "image_width": metadata.image_size["width"],
                "image_height": metadata.image_size["height"]
            })

        csv_file = os.path.join(output_dir, "figures_mapping.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"✓ 映射表已保存: {csv_file}")

        # 总结
        result = {
            "pdf_name": pdf_name,
            "pdf_path": os.path.abspath(pdf_path),
            "total_pages": len(page_images),
            "total_figures": len(all_metadata),
            "output_directory": output_dir,
            "figures": [asdict(m) for m in all_metadata],
            "processed_at": datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"处理完成！")
        print(f"总页数: {result['total_pages']}")
        print(f"提取图片数: {result['total_figures']}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")

        return result


def main():
    """主函数"""
    # 配置
    API_KEY = "sk-6a313d7b10cf4c9fa525708c8fadf0d1"
    PDF_PATH = r"D:\shixixiangmu\pdf\data\semi_final_dataset\semi_final_dataset\train\documents\CN1022735C.pdf"
    OUTPUT_DIR = r"D:\shixixiangmu\pdf\figure_extraction_output"

    # 初始化检测器
    detector = QwenVLFigureDetector(api_key=API_KEY)

    # 初始化提取器
    extractor = PDFFigureExtractor(detector)

    # 处理 PDF
    result = extractor.process_pdf(PDF_PATH, OUTPUT_DIR, dpi=600)

    # 保存处理结果
    result_file = os.path.join(OUTPUT_DIR, "processing_result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ 处理结果已保存: {result_file}")


if __name__ == "__main__":
    main()
