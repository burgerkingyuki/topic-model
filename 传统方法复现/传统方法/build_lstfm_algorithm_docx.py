from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = r"E:\主题建模\LSTFM主题建模\传统方法\LSTFM_algorithm.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tc_pr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tc_pr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        if edge in kwargs:
            edge_data = kwargs.get(edge)
            tag = 'w:{}'.format(edge)
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            for key in ['sz', 'val', 'color', 'space']:
                if key in edge_data:
                    element.set(qn('w:{}'.format(key)), str(edge_data[key]))


def set_table_borders(table):
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(
                cell,
                top={'val': 'single', 'sz': '6', 'color': 'D9E2EC'},
                bottom={'val': 'single', 'sz': '6', 'color': 'D9E2EC'},
                left={'val': 'single', 'sz': '6', 'color': 'D9E2EC'},
                right={'val': 'single', 'sz': '6', 'color': 'D9E2EC'},
            )


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in('w:tcMar')
    if tc_mar is None:
        tc_mar = OxmlElement('w:tcMar')
        tc_pr.append(tc_mar)
    for m, v in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tc_mar.find(qn(f'w:{m}'))
        if node is None:
            node = OxmlElement(f'w:{m}')
            tc_mar.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement('w:tblHeader')
    tbl_header.set(qn('w:val'), 'true')
    tr_pr.append(tbl_header)


def set_cell_width(cell, width_twips):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn('w:tcW'))
    if tc_w is None:
        tc_w = OxmlElement('w:tcW')
        tc_pr.append(tc_w)
    tc_w.set(qn('w:w'), str(width_twips))
    tc_w.set(qn('w:type'), 'dxa')


def add_run(paragraph, text, bold=False, italic=False, size=9.5, color='111111', font='Calibri'):
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = font
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    return run


def fill_paragraph(paragraph, text, bold=False, size=9.5, color='111111'):
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.05
    add_run(paragraph, text, bold=bold, size=size, color=color)


def set_fixed_table_layout(table, widths):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    layout = tbl_pr.find(qn('w:tblLayout'))
    if layout is None:
        layout = OxmlElement('w:tblLayout')
        tbl_pr.append(layout)
    layout.set(qn('w:type'), 'fixed')
    tbl_w = tbl_pr.find(qn('w:tblW'))
    if tbl_w is None:
        tbl_w = OxmlElement('w:tblW')
        tbl_pr.append(tbl_w)
    tbl_w.set(qn('w:w'), str(sum(widths)))
    tbl_w.set(qn('w:type'), 'dxa')
    grid = tbl.tblGrid
    if grid is None:
        grid = OxmlElement('w:tblGrid')
        tbl.append(grid)
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        grid_col = OxmlElement('w:gridCol')
        grid_col.set(qn('w:w'), str(width))
        grid.append(grid_col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, widths[idx])


def main():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    styles = doc.styles
    normal = styles['Normal']
    normal.font.name = 'Calibri'
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimSun')
    normal.font.size = Pt(10)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    add_run(title, 'Algorithm 1  LSTFM', bold=True, size=13, color='0B2545')

    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [780, 8580]
    set_fixed_table_layout(table, widths)
    set_table_borders(table)

    rows = [
        ('Input:', 'D={d_i}_{i=1}^{N}, S={s_i}_{i=1}^{N}, T_1^{seed}, K^{seed}, K_{wiki}, K_{local}'),
        ('Output:', 'T^*=(T_1^*,T_2^*)'),
        ('Parameters:', 'LLM is the generation and refinement model; E_1 is the topic embedding model; E_2 is the evidence retrieval embedding model; tau is the merging threshold; M is the first-level topic limit; n_min is the minimum document number for clustering; R_max is the maximum retry number for rejected topics.'),
        ('1:', 'T <- T_1^{seed}, T_2^* <- emptyset'),
        ('2:', 'for each d_i in D do'),
        ('3:', '    C_{1,i} <- GenerateFirstLevel(d_i,T_1^{seed},T,LLM)'),
        ('4:', '    T <- Update_1(T,C_{1,i})'),
        ('5:', 'end for'),
        ('6:', 'T <- MergeFirstLevel(T,E_1,LLM,tau,M)'),
        ('7:', 'for each d_i in D do'),
        ('8:', '    C_{2,i} <- GenerateSecondLevel(s_i,C_{1,i},T,LLM)'),
        ('9:', '    T <- Update_2(T,C_{2,i})'),
        ('10:', 'end for'),
        ('11:', 'for each t in T_1 do'),
        ('12:', '    R_t <- ClusterKeywords(T_2,t,E_1,n_min)'),
        ('13:', '    for each r in R_t do'),
        ('14:', '        e_r <- RetrieveAndRefine(t,r,K_{wiki},K_{local},E_2,LLM)'),
        ('15:', '        k <- 0, y <- REJECT, rho <- emptyset'),
        ('16:', '        while y=REJECT and k<R_max do'),
        ('17:', '            z_r <- Standardize(t,r,K^{seed},e_r,rho,LLM)'),
        ('18:', '            (y,rho) <- Validate(z_r,Parent(z_r),Evidence(z_r))'),
        ('19:', '            k <- k+1'),
        ('20:', '        end while'),
        ('21:', '        if y=ACCEPT then'),
        ('22:', '            T_2^* <- T_2^* union {z_r}'),
        ('23:', '        end if'),
        ('24:', '    end for'),
        ('25:', 'end for'),
        ('26:', 'return T^*=(T_1^*,T_2^*)'),
    ]

    # Reuse first row then add remaining rows.
    for row_idx, (label, content) in enumerate(rows):
        if row_idx == 0:
            row = table.rows[0]
        else:
            row = table.add_row()
        set_repeat_table_header(row) if row_idx == 0 else None
        for idx, text in enumerate((label, content)):
            cell = row.cells[idx]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_width(cell, widths[idx])
            p = cell.paragraphs[0]
            if idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if label.endswith(':') and label[0].isdigit() else WD_ALIGN_PARAGRAPH.LEFT
                fill_paragraph(p, text, bold=(row_idx < 3), size=9.2, color='0B2545' if row_idx < 3 else '222222')
            else:
                fill_paragraph(p, text, bold=False, size=9.2, color='111111')
        if row_idx < 3:
            set_cell_shading(row.cells[0], 'E8EEF5')
            set_cell_shading(row.cells[1], 'F7F9FC')

    # Make the outer box slightly more algorithm-like with top/bottom rules.
    for cell in table.rows[0].cells:
        set_cell_border(cell, top={'val': 'single', 'sz': '14', 'color': '0B2545'}, bottom={'val': 'single', 'sz': '6', 'color': 'D9E2EC'}, left={'val': 'single', 'sz': '6', 'color': 'D9E2EC'}, right={'val': 'single', 'sz': '6', 'color': 'D9E2EC'})
    for cell in table.rows[-1].cells:
        set_cell_border(cell, bottom={'val': 'single', 'sz': '14', 'color': '0B2545'}, top={'val': 'single', 'sz': '6', 'color': 'D9E2EC'}, left={'val': 'single', 'sz': '6', 'color': 'D9E2EC'}, right={'val': 'single', 'sz': '6', 'color': 'D9E2EC'})

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(7)
    note.paragraph_format.space_after = Pt(0)
    add_run(note, 'Note: Symbols are kept as editable linear formula text for convenient conversion in Word.', italic=True, size=8.5, color='555555')

    doc.save(OUT)


if __name__ == '__main__':
    main()
