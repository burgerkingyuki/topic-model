# -*- coding: utf-8 -*-
"""Build a three-line (booktabs-style) Word table of reproduced baseline parameters.

Reads nothing at run time: values below were transcribed from each
config.json under 传统方法复现/, with K unified to 40 as agreed.
"""

import os

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = r"E:\主题建模\LSTFM\传统方法复现\复现基线模型参数设置表.docx"

LATIN = "Times New Roman"
CJK = "宋体"

HEADERS = ["模型", "数据集", "主题数 K", "参数", "取值"]
# total 9648 twips = 6.7 in, exactly the text width of the page below
WIDTHS = [1700, 1000, 950, 3728, 2270]

DATASETS = "ROBOT / ICLR / Q"

GROUPS = [
    ("LDA", DATASETS, "40", [
        ("passes（语料遍历轮数）", "10"),
        ("alpha（文档–主题先验）", "auto"),
        ("no_below（词典最小文档频次）", "2"),
        ("no_above（词典最大文档频率比例）", "0.8"),
        ("top_words_per_topic（每主题输出词数）", "10"),
        ("random_state（随机种子）", "42"),
    ]),
    ("NMF", DATASETS, "40", [
        ("init（初始化方式）", "nndsvd"),
        ("max_iter（最大迭代次数）", "1000"),
        ("tol（收敛容差）", "0.0001"),
        ("alpha_w（正则化权重）", "0.0"),
        ("l1_ratio（L1 正则比例）", "0.01"),
        ("max_features（词表规模上限）", "5000"),
        ("min_df（最小文档频次）", "2"),
        ("max_df（最大文档频率比例）", "0.9"),
        ("top_words_per_topic（每主题输出词数）", "5"),
        ("random_state（随机种子）", "42"),
    ]),
    ("BERTopic", DATASETS, "40", [
        ("umap_n_neighbors（UMAP 邻域数）", "15"),
        ("umap_n_components（UMAP 降维维度）", "5"),
        ("umap_min_dist（UMAP 最小间距）", "0.0"),
        ("hdbscan_min_cluster_size（最小簇规模）", "5"),
        ("hdbscan_min_samples（核心点最小样本数）", "1"),
        ("hdbscan_cluster_selection_epsilon（簇选择半径）", "0.1"),
        ("vectorizer_min_df（c-TF-IDF 最小文档频次）", "1"),
        ("vectorizer_max_df（最大文档频率比例）", "0.95"),
        ("vectorizer_max_features（词表规模上限）", "5000"),
        ("top_n_words（每主题输出词数）", "10"),
    ]),
    ("PromptTopic", DATASETS, "40", [
        ("generation_mode（主题生成模式）", "full_text"),
        ("demonstrations（few-shot 演示样例数）", "2"),
        ("max_document_chars（文档截断字符数）", "12000"),
        ("max_topics_per_document（单文档生成主题上限）", "1"),
        ("intermediate_topics（中间压缩主题数）", "400"),
        ("collapse_strategy（主题坍缩策略）", "hybrid"),
        ("pbm_window_size（PBM 匹配滑动窗口）", "50"),
        ("ctfidf_top_n（c-TF-IDF 表征词数）", "100"),
        ("representation_top_n（主题表征候选词数）", "10"),
        ("use_llm_word_filter（是否用 LLM 过滤主题词）", "是"),
        ("batch_size（批处理大小）", "16"),
        ("random_seed（随机种子）", "42"),
    ]),
    ("TopicGPT", DATASETS, "自动\n确定", [
        ("generation_mode（主题生成模式）", "full_text"),
        ("sample_size（抽样文档数）", "1000"),
        ("max_topic_rounds（生成—修正轮次上限）", "3"),
        ("merge_similarity_threshold（主题合并阈值）", "0.5"),
        ("low_frequency_removal_threshold（低频主题剔除阈值）", "3"),
        ("max_assignments_per_doc（单文档分配主题上限）", "1"),
        ("assignment_self_correction_retries（分配自纠错重试次数）", "10"),
        ("batch_size（批处理大小）", "16"),
    ]),
    ("LISA", DATASETS, "40", [
        ("document_cluster_method（文档聚类方法）", "kmeans"),
        ("topic_cluster_method（主题聚类方法）", "kmeans"),
        ("generation_mode（主题生成模式）", "full_text"),
        ("max_document_chars（文档截断字符数）", "12000"),
        ("lambda_neighbors（λ 近邻数）", "5"),
        ("mu_neighbors（μ 近邻数）", "15"),
        ("hidden_dims（TPN 隐层维度）", "[256, 256]"),
        ("train_epochs（训练轮数）", "20"),
        ("train_batch_size（训练批大小）", "128"),
        ("tpn_learning_rate（学习率）", "0.001"),
        ("tau（温度系数 τ）", "1.5"),
        ("alpha（分配权重 α）", "0.8"),
        ("low_confidence_threshold（低置信度阈值）", "0.55"),
        ("random_seed（随机种子）", "42"),
    ]),
]

CAPTION = "表 1  复现基线模型的主要参数设置"

NOTES = [
    "注：① ROBOT、ICLR、Q 三个数据集使用同一套参数；各方法的输入均为标准化 JSONL 的 text 全文字段。",
    "② 为与本文方法公平对比，所有受主题数约束的基线统一取主题数 K = 40；TopicGPT 不预设主题数，"
    "其最终主题数由生成—修正轮次（max_topic_rounds）、合并阈值（merge_similarity_threshold）"
    "与低频主题剔除阈值（low_frequency_removal_threshold）共同决定。",
    "③ 需要语义表征的模型统一使用 all-MiniLM-L12-v2 句向量模型；LLM 相关方法统一调用 gpt-4o-mini。",
    "④ 评价口径统一：主题词取前 10 个（top_n = 10），一致性滑窗为 110，"
    "文档代表性评测每篇最多抽取 12 句、每句不少于 20 字符。",
    "⑤ PromptTopic 的 few-shot 演示样例数为 2：原文报告 N = 4 时效果最佳，但论文仅部分披露示例提示词，"
    "故按可获取的示例配置；主题坍缩采用 PBM→WSM（hybrid）策略。",
]

SOURCE = (
    "参数来源：LDA、NMF、BERTopic 见 传统方法/新版本/<模型>/config.json；"
    "PromptTopic、TopicGPT、LISA 见 LLM-based/<模型>/config.json；"
    "评价口径见 LLM-based/metrics_compute/config.json。"
)


# ---------------------------------------------------------------- xml helpers
def _el(tag, **attrs):
    node = OxmlElement(tag if ":" in tag else "w:" + tag)
    for key, value in attrs.items():
        node.set(qn("w:" + key), str(value))
    return node


def set_cell_edge(cell, name, sz, color="000000"):
    """Draw one horizontal rule on a single cell edge (sz is in 1/8 pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    found = tc_pr.find(qn("w:tcBorders"))
    if found is None:
        found = OxmlElement("w:tcBorders")
        # schema order: tcW, gridSpan, hMerge, vMerge, tcBorders, shd, noWrap, tcMar, ...
        anchor = None
        for tag in ("w:shd", "w:noWrap", "w:tcMar", "w:textDirection",
                    "w:tcFitText", "w:vAlign", "w:hideMark"):
            anchor = tc_pr.find(qn(tag))
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addprevious(found)
        else:
            tc_pr.append(found)
    for old in found.findall(qn("w:" + name)):
        found.remove(old)
    found.append(_el(name, val="single", sz=sz, space=0, color=color))


def set_table_borders(table, top_sz=12, mid_sz=6, bottom_sz=12, color="000000"):
    """Three-line table: clear every edge, then draw three rules."""
    tbl_pr = table._tbl.tblPr
    for old in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(old)
    borders = _el("tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_el(edge, val="nil", sz=0, space=0, color="auto"))
    # schema order: ... tblInd, tblBorders, shd, tblLayout, tblCellMar, tblLook
    anchor = tbl_pr.find(qn("w:tblLayout"))
    if anchor is None:
        anchor = tbl_pr.find(qn("w:tblLook"))
    if anchor is not None:
        anchor.addprevious(borders)
    else:
        tbl_pr.append(borders)

    for cell in table.rows[0].cells:
        set_cell_edge(cell, "top", top_sz, color)
        set_cell_edge(cell, "bottom", mid_sz, color)
    for cell in table.rows[-1].cells:
        set_cell_edge(cell, "bottom", bottom_sz, color)


def add_group_rules(table, start_rows, sz=6, color="000000"):
    """Separator rule between adjacent model blocks, spanning all columns."""
    for row_index in start_rows:
        for cell in table.rows[row_index].cells:
            set_cell_edge(cell, "top", sz, color)


def set_cell_margins(cell, top=40, start=80, bottom=40, end=80):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.find(qn("w:tcMar"))
    if mar is None:
        mar = OxmlElement("w:tcMar")
        anchor = None
        for tag in ("w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark"):
            anchor = tc_pr.find(qn(tag))
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addprevious(mar)
        else:
            tc_pr.append(mar)
    for name, value in (("top", top), ("left", start), ("bottom", bottom), ("right", end)):
        for old in mar.findall(qn("w:" + name)):
            mar.remove(old)
        mar.append(_el(name, w=value, type="dxa"))


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    # schema order: cantSplit, trHeight, tblHeader
    tr_pr.append(_el("tblHeader", val="true"))


def set_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    # schema order: cantSplit, trHeight, tblHeader -> insert before tblHeader
    header = tr_pr.find(qn("w:tblHeader"))
    node = _el("cantSplit")
    if header is not None:
        header.addprevious(node)
    else:
        tr_pr.append(node)


def set_fixed_layout(table, widths):
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        grid.append(_el("gridCol", w=width))
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            for old in tc_pr.findall(qn("w:tcW")):
                tc_pr.remove(old)
            tc_pr.append(_el("tcW", w=widths[idx], type="dxa"))


# ------------------------------------------------------------- text helpers
def style_run(run, size=9, bold=False, color="000000", italic=False):
    run.font.name = LATIN
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:ascii"), LATIN)
    r_fonts.set(qn("w:hAnsi"), LATIN)
    r_fonts.set(qn("w:eastAsia"), CJK)


def write_cell(cell, text, align=WD_ALIGN_PARAGRAPH.CENTER, bold=False,
               size=9, color="000000"):
    lines = text.split("\n")
    cell.text = ""
    paragraph = cell.paragraphs[0]
    for index, line in enumerate(lines):
        if index:
            paragraph.add_run().add_break()
        style_run(paragraph.add_run(line), size=size, bold=bold, color=color)
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = 1.0
    paragraph.alignment = align
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def write_paragraph(doc, text, size=8, bold=False, italic=False,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY, color="000000",
                    space_before=0, space_after=0, first_line_indent=None):
    paragraph = doc.add_paragraph()
    style_run(paragraph.add_run(text), size=size, bold=bold, italic=italic, color=color)
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    fmt.line_spacing = 1.15
    paragraph.alignment = align
    if first_line_indent is not None:
        fmt.first_line_indent = Inches(first_line_indent)
    return paragraph


# ------------------------------------------------------------------- builder
def main(out_path=OUT, wide=False):
    doc = Document()

    section = doc.sections[0]
    if wide:
        # Diagnostic copy only: a page wide enough that the bundled renderer's
        # 1.5x-zoom crop still shows the whole table. Column widths are fixed,
        # so cell wrapping is identical to the delivered document.
        section.page_width = Inches(12)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.4)
        section.bottom_margin = Inches(0.4)
        section.left_margin = Inches(0.4)
        section.right_margin = Inches(0.4)
    else:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = LATIN
    normal.font.size = Pt(9)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), CJK)

    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(0)
    caption.paragraph_format.space_after = Pt(4)
    style_run(caption.add_run(CAPTION), size=10.5, bold=True)

    total_rows = 1 + sum(len(group[3]) for group in GROUPS)
    table = doc.add_table(rows=total_rows, cols=len(HEADERS))
    # The delivered table is as wide as the text column, so centring and left
    # alignment coincide; the diagnostic copy is left aligned to stay visible.
    table.alignment = WD_TABLE_ALIGNMENT.LEFT if wide else WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_fixed_layout(table, WIDTHS)

    set_repeat_header(table.rows[0])
    set_cant_split(table.rows[0])
    for idx, header in enumerate(HEADERS):
        cell = table.rows[0].cells[idx]
        set_cell_margins(cell)
        write_cell(cell, header, bold=True, size=9)

    row_index = 1
    group_starts = []
    for model, dataset, k_value, params in GROUPS:
        start = row_index
        group_starts.append(start)
        for param, value in params:
            row = table.rows[row_index]
            set_cant_split(row)
            for cell in row.cells:
                set_cell_margins(cell)
            write_cell(row.cells[3], param, align=WD_ALIGN_PARAGRAPH.LEFT, size=9)
            write_cell(row.cells[4], value, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
            row_index += 1
        end = row_index - 1

        for col, text in ((0, model), (1, dataset), (2, k_value)):
            merged = table.cell(start, col)
            if end > start:
                merged = merged.merge(table.cell(end, col))
            set_cell_margins(merged)
            write_cell(merged, text, size=9, bold=(col == 0))

        # Keep every model block on one page: a vertically merged model cell
        # cannot carry its text across a page break, so it would render empty.
        for r in range(start, end):
            for cell in table.rows[r].cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True

    set_table_borders(table)
    add_group_rules(table, group_starts[1:])

    for note in NOTES:
        write_paragraph(doc, note, size=8, space_before=2, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    write_paragraph(doc, SOURCE, size=7.5, italic=True, color="595959",
                    space_before=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
    print("saved:", out_path)
    print("rows:", total_rows, "cols:", len(HEADERS))


if __name__ == "__main__":
    main()
    main(out_path=os.path.join(os.path.dirname(OUT), "scripts", "diag_wide.docx"), wide=True)
