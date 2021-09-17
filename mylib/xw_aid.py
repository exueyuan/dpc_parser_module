# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>
"""
Excel输出辅助
"""
import math
import copy
import xlrd
import xlwt
from xlwt.Style import default_style


MaxColCount = 256


def get_cell(sheet, row_num, col_num, as_str=False):
    value = sheet.cell(row_num, col_num).value
    if as_str and not isinstance(value, str):
        value = "" if isinstance(value, float) and not math.isnan(value) else str(value)
    return value


def load_xls_sheet(file_name, sheet=0, col_count=0, as_str=False):
    with xlrd.open_workbook(file_name) as xls_book:
        sheet = xls_book.sheets()[sheet]
        col_count = col_count if col_count > 0 else sheet.ncols
        data_row_list = []
        for row_num in range(0, sheet.nrows):
            row_cells = [get_cell(sheet, row_num, ci, as_str) for ci in range(col_count)]
            data_row_list.append(row_cells)
        return data_row_list


def new_sheet(sheet_name="sheet1", row_default_height=300):
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(sheet_name)
    worksheet.row_default_height = row_default_height
    return workbook, worksheet


def add_sheet(workbook, sheet_name, row_default_height=300):
    worksheet = workbook.add_sheet(sheet_name)
    worksheet.row_default_height = row_default_height
    return worksheet


def create_text_style(align_center=False, wordwrap=False, font_color=None):
    align = xlwt.Alignment()
    align.horz = xlwt.Alignment.HORZ_CENTER if align_center else xlwt.Alignment.HORZ_LEFT
    align.vert = xlwt.Alignment.VERT_CENTER
    if wordwrap:
        align.wrap = xlwt.Alignment.WRAP_AT_RIGHT
    style = xlwt.XFStyle()
    style.alignment = align
    if font_color is not None:
        font = xlwt.Font()
        font.colour_index = font_color
        style.font = font

    return style


def create_number_style(align_center=False):
    align = xlwt.Alignment()
    align.horz = xlwt.Alignment.HORZ_CENTER if align_center else xlwt.Alignment.HORZ_RIGHT
    align.vert = xlwt.Alignment.VERT_CENTER
    style = xlwt.XFStyle()
    style.alignment = align
    style.num_format_str = "String"
    return style


color_green = 17
color_purple = 32
color_gray = 23
color_maroon = 25
color_navy = 18
color_pink = 45


def change_font_color(font_color, copy_style=None):
    if copy_style is None:
        style = xlwt.XFStyle()
    else:
        style = copy.deepcopy(copy_style)

    style.font.colour_index = font_color
    return style


def set_font_style(bold, italic, copy_style=None):
    if copy_style is None:
        style = xlwt.XFStyle()
    else:
        style = copy.deepcopy(copy_style)

    style.font.bold = bold
    style.font.italic = italic
    return style


def set_col_widths(worksheet, estimate_col_chars):
    estimate_col_chars = estimate_col_chars[:MaxColCount]
    for col_num, show_chars in enumerate(estimate_col_chars):
        if show_chars > 0:
            worksheet.col(col_num).width = show_chars * 300


def merge(worksheet, top, bottom, left, right, text):
    worksheet.write_merge(top, bottom, left, right, text)


def set_cell(worksheet, row_num, col_num, text, style):
    worksheet.write(row_num, col_num, text, style)


def set_row_cells(worksheet, row_num, cell_values, col_styles=None, skip_zero=False):
    cell_values = cell_values[:MaxColCount]
    for col_num, value in enumerate(cell_values):
        if value is None:
            continue
        style = default_style
        if col_styles is not None and col_num < len(col_styles):
            style = col_styles[col_num]
            if style is None:
                style = default_style
        if skip_zero and (isinstance(value, int) and value == 0 or isinstance(value, float) and value == 0.0):
            continue
        worksheet.write(row_num, col_num, value, style)
