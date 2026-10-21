#!/usr/bin/env python3
"""
Markdown Table Parser for UHAMKA Academic Documents
Converts messy markdown tables from 6 curriculum docs into clean CSVs.

Priority tables:
1. mata_kuliah (course list per semester)
2. semester_summary (totals rows)
3. mata_kuliah_pengayaan (enrichment courses)
4. program_studi (program identity, optional)
5. cpl (learning outcomes, optional)
6. cpl_mata_kuliah (CPL-course matrix, optional)
"""

import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TableRow:
    cells: List[str]
    line_num: int
    is_separator: bool = False

@dataclass
class TableBlock:
    rows: List[TableRow]
    start_line: int
    end_line: int
    prodi: str
    source_file: str

# Prodi mapping
PRODI_MAP = {'Teknik Informatika': 'TI', 'TRM': 'TRM', 'TE': 'TE', 'STI': 'STI', 'TM': 'TM'}
KURIKULUM_MAP = {'TRM': '2024', 'TE': '2023-2024', 'TI': '2025-2026', 'STI': '2025-2026', 'TM': '2026-2027'}

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def is_separator_row(cells: List[str]) -> bool:
    return all(re.match(r'^-+$', c.strip()) for c in cells if c.strip())

def is_empty_row(cells: List[str]) -> bool:
    return all(not c.strip() for c in cells)

def is_banner_row(cells: List[str]) -> bool:
    if not cells or not cells[0].strip():
        return False
    return all(not c.strip() for c in cells[1:])

def is_totals_row(cells: List[str]) -> bool:
    first_cell = cells[0].strip().lower() if cells else ''
    return any(kw in first_cell for kw in ['jumlah', 'total'])

def is_reject_table(first_cells: List[str]) -> bool:
    text = ' '.join(first_cells[:3]).lower()
    reject_patterns = ['daftar isi', 'tabel_placeholder', 'tabel_broadheader',
                       'logo:', 'icon:', 'error! bookmark', 'identitas program',
                       'rps', '<hr/>', 'panduan akademik ta.']
    return any(pat in text for pat in reject_patterns)

def extract_prodi_from_filename(filename: str) -> str:
    fname = Path(filename).stem.lower()
    if 'informatika' in fname:
        return 'TI'
    elif 'trm' in fname or 'mekatronika' in fname:
        return 'TRM'
    elif 'sti' in fname:
        return 'STI'
    elif 'tm-panduan' in fname or 'mesin' in fname:
        return 'TM'
    elif 'prodi te' in fname or 'elektro' in fname:
        return 'TE'
    elif 'pengayaan' in fname:
        return 'PENGAYAAN'
    return 'UNKNOWN'

def parse_markdown_table(lines: List[str], start_idx: int, filename: str) -> Optional[TableBlock]:
    rows = []
    current_idx = start_idx
    prodi = extract_prodi_from_filename(filename)

    while current_idx < len(lines):
        line = lines[current_idx]
        stripped = line.strip()
        if not stripped.startswith('|'):
            break
        cells = [normalize_whitespace(c) for c in stripped.split('|')[1:-1]]
        if not cells or is_empty_row(cells):
            current_idx += 1
            continue
        is_sep = is_separator_row(cells)
        rows.append(TableRow(cells=cells, line_num=current_idx + 1, is_separator=is_sep))
        current_idx += 1

    if not rows:
        return None
    first_content = [r.cells for r in rows if not r.is_separator][:3]
    if is_reject_table([c for row in first_content for c in row]):
        return None

    return TableBlock(rows=rows, start_line=start_idx + 1, end_line=current_idx,
                      prodi=prodi, source_file=Path(filename).name)

def clean_leading_empty_column(block: TableBlock) -> None:
    data_rows = [r for r in block.rows if not r.is_separator]
    if not data_rows:
        return
    if all(not r.cells[0].strip() for r in data_rows):
        for row in block.rows:
            if row.cells:
                row.cells = row.cells[1:]

SEM_HEADING_RE = re.compile(r'SEMESTER[\s\-–]+(VIII|VII|VI|IV|IX|V|III|II|I|\d+)', re.IGNORECASE)

def merge_page_split_tables(blocks: List[TableBlock],
                            lines_by_file: Optional[Dict[str, List[str]]] = None) -> List[TableBlock]:
    if not blocks:
        return []
    merged = [blocks[0]]
    for current in blocks[1:]:
        prev = merged[-1]
        if current.prodi == prev.prodi and current.start_line - prev.end_line < 10:
            # Never merge across a semester boundary: consecutive per-semester
            # tables have identical headers, so the semester heading between
            # them is the only thing separating "Semester III" from "Semester V".
            between: List[str] = []
            if lines_by_file:
                all_lines = lines_by_file.get(current.source_file, [])
                between = all_lines[prev.end_line:current.start_line - 1]
            if any(SEM_HEADING_RE.search(ln) for ln in between):
                merged.append(current)
                continue
            prev_header = next((r.cells for r in prev.rows if not r.is_separator), None)
            curr_header = next((r.cells for r in current.rows if not r.is_separator), None)
            if prev_header and curr_header:
                ph = tuple(normalize_whitespace(c).lower() for c in prev_header)
                ch = tuple(normalize_whitespace(c).lower() for c in curr_header)
                if ph == ch:
                    data_rows = [r for r in current.rows if not r.is_separator][1:]
                    prev.rows.extend(data_rows)
                    prev.end_line = current.end_line
                    continue
        merged.append(current)
    return merged

def extract_semester_from_context(blocks: List[TableBlock], block_idx: int, lines: List[str]) -> Optional[int]:
    roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}
    block = blocks[block_idx]

    def parse(match: 're.Match') -> Optional[int]:
        sem_str = match.group(1).upper()
        return roman_map.get(sem_str, int(sem_str) if sem_str.isdigit() else None)

    # An in-table "SEMESTER X" banner row is the most specific source of truth.
    for row in [r for r in block.rows if not r.is_separator][:6]:
        if is_banner_row(row.cells):
            m = SEM_HEADING_RE.search(row.cells[0])
            if m:
                value = parse(m)
                if value:
                    return value
    if lines:
        # Then the nearest heading above, then headings within the first lines below.
        for i in range(block.start_line - 1, max(0, block.start_line - 26), -1):
            m = SEM_HEADING_RE.search(lines[i])
            if m:
                value = parse(m)
                if value:
                    return value
        for i in range(block.start_line, min(len(lines), block.start_line + 8)):
            m = SEM_HEADING_RE.search(lines[i])
            if m:
                value = parse(m)
                if value:
                    return value
    return None

def _header_cells(block: TableBlock) -> Optional[List[str]]:
    """First non-separator, non-banner row, merged with a grouped second header.

    TRM-style tables have "| ... | Bobot sks | | |" followed by
    "| | Teori | Praktikum | Jumlah |": the export shifts the sub-names, so
    place them sequentially starting at the "Bobot sks"/SKS group column.
    """
    content = [r.cells for r in block.rows if not r.is_separator and not is_banner_row(r.cells)]
    if not content:
        return None
    header = list(content[0])
    if len(content) > 1:
        second = content[1]
        if second and not re.match(r'^\d+$', second[0].strip()):
            sub_names = [c for c in second
                         if any(kw in c.lower() for kw in ('teori', 'praktikum', 'praktek', 'pengayaan', 'jumlah'))]
            group_idx = next((i for i, c in enumerate(header)
                              if 'bobot' in c.lower() or re.search(r'\bsks\b', c.lower())), None)
            if sub_names and group_idx is not None:
                for k, name in enumerate(sub_names):
                    if group_idx + k < len(header):
                        header[group_idx + k] = name
    return header

def is_course_table(block: TableBlock) -> bool:
    if not block.rows:
        return False
    header = _header_cells(block)
    if not header:
        return False
    header_text = ' '.join(header).lower()
    has_kode = any(kw in header_text for kw in ['kode mk', 'kode mata kuliah', 'kode matakuliah'])
    has_nama = any(kw in header_text for kw in ['mata kuliah', 'nama mata', 'nama mk'])
    has_sks = any(kw in header_text for kw in ['sks', 'bobot', 'teori', 'praktikum', 'praktek', 'jumlah'])
    return has_kode and has_nama and has_sks

def is_enrichment_table(block: TableBlock) -> bool:
    if not block.rows:
        return False
    header_row = next((r for r in block.rows if not r.is_separator), None)
    if not header_row:
        return False
    header_text = ' '.join(header_row.cells).lower()
    return 'pengayaan' in header_text or 'berpikir kritis' in ' '.join(
        [' '.join(r.cells).lower() for r in block.rows[:5]])

SKS_RE = re.compile(r'^\d{1,2}$')
KODE_RE = re.compile(r'^0?\d{6,8}$')

def parse_course_row(cells: List[str], prodi: str, header: Optional[List[str]] = None) -> Optional[Dict]:
    """Parse one course row using the table's header to locate the columns.

    Headers vary per document: [No|Kode|Nama|SKS|Prasyarat], [No|Kode|Nama|Kode CPMK|SKS|CPL...],
    [No|Kode|Nama|Teori|Praktikum|Jumlah] ... Mapping by column name avoids the
    positional misalignment that put group codes into the SKS field.
    """
    norm = [normalize_whitespace(c) for c in cells]
    if not norm or is_totals_row(norm) or is_banner_row(norm):
        return None

    kode_idx = nama_idx = sks_idx = teori_idx = praktikum_idx = praktek_idx = sem_idx = prasyarat_idx = None
    if header:
        hl = [normalize_whitespace(c).lower() for c in header]
        for i, h in enumerate(hl):
            if kode_idx is None and 'kode' in h:
                kode_idx = i
            if nama_idx is None and 'kode' not in h and any(k in h for k in ('nama', 'mata kuliah', 'matakuliah')):
                nama_idx = i
            if sks_idx is None and re.search(r'\bsks\b|jumlah|bobot', h) \
                    and not any(k in h for k in ('teori', 'praktikum', 'praktek')):
                sks_idx = i
            if teori_idx is None and 'teori' in h:
                teori_idx = i
            if praktikum_idx is None and 'praktikum' in h:
                praktikum_idx = i
            if praktek_idx is None and 'praktek' in h:
                praktek_idx = i
            if sem_idx is None and re.match(r'^(smt\.?|sem\.?|semester)$', h.replace(' ', '')):
                sem_idx = i
            if prasyarat_idx is None and 'prasyarat' in h:
                prasyarat_idx = i

    if kode_idx is None or kode_idx >= len(norm) or not KODE_RE.match(norm[kode_idx]):
        kode_idx = next((i for i, c in enumerate(norm) if KODE_RE.match(c)), None)
        if kode_idx is None:
            return None
        nama_idx = None
    kode_mk = norm[kode_idx]

    if nama_idx is None or nama_idx >= len(norm) or not norm[nama_idx].strip():
        nama_idx = kode_idx + 1
    while nama_idx < len(norm) and not norm[nama_idx].strip():
        nama_idx += 1
    if nama_idx >= len(norm) or KODE_RE.match(norm[nama_idx]):
        return None
    nama_mk = norm[nama_idx]

    def cell(idx: Optional[int]) -> Optional[str]:
        if idx is None or idx >= len(norm):
            return None
        value = norm[idx].strip()
        return value or None

    sks = cell(sks_idx)
    if sks and not SKS_RE.match(sks):
        sks = None
    teori, praktikum, praktek = cell(teori_idx), cell(praktikum_idx), cell(praktek_idx)
    for name, value in (('teori', teori), ('praktikum', praktikum), ('praktek', praktek)):
        if value and not SKS_RE.match(value):
            if name == 'teori':
                teori = None
            elif name == 'praktikum':
                praktikum = None
            else:
                praktek = None
    if sks is None and any((teori, praktikum, praktek)):
        nums = [int(v) for v in (teori, praktikum, praktek) if v and v.isdigit()]
        sks = str(sum(nums)) if nums else None
    if sks is None:
        for i in range(kode_idx + 2, min(len(norm), kode_idx + 7)):
            if SKS_RE.match(norm[i]) and 1 <= int(norm[i]) <= 12:
                sks = norm[i]
                break

    semester = None
    sem_raw = cell(sem_idx)
    if sem_raw:
        nums = set(re.findall(r'\d+', sem_raw))
        if len(nums) == 1:
            value = int(nums.pop())
            if 1 <= value <= 14:
                semester = value

    return {'kode_mk': kode_mk, 'nama_mk': nama_mk, 'sks': sks,
            'sks_teori': teori, 'sks_praktikum': praktikum, 'sks_praktek': praktek,
            'prasyarat': cell(prasyarat_idx), 'semester': semester,
            'sem_explicit': sem_idx is not None}

def parse_totals_row(cells: List[str]) -> Optional[Dict]:
    if not is_totals_row(cells):
        return None
    numeric_cells = [c.strip() for c in cells[1:] if c.strip() and c.strip().isdigit()]
    if not numeric_cells:
        return None
    summary = {'sks_dokumen': int(numeric_cells[-1])}
    if len(numeric_cells) >= 3:
        summary['sks_teori'] = int(numeric_cells[0]) if numeric_cells[0].isdigit() else None
        summary['sks_praktikum'] = int(numeric_cells[1]) if numeric_cells[1].isdigit() else None
        summary['sks_praktek'] = int(numeric_cells[2]) if len(numeric_cells) > 2 else None
    return summary

def build_mata_kuliah_csv(blocks: List[TableBlock], lines_by_file: Dict[str, List[str]], output_path: str):
    courses: Dict[tuple, Dict] = {}
    order: List[tuple] = []
    for block_idx, block in enumerate(blocks):
        if not is_course_table(block):
            continue
        clean_leading_empty_column(block)
        semester = extract_semester_from_context(blocks, block_idx,
                                                  lines_by_file.get(block.source_file, []))
        header = _header_cells(block)
        for row in block.rows:
            if row.is_separator:
                continue
            course = parse_course_row(row.cells, block.prodi, header)
            if not course:
                continue
            # A row-level semester column (e.g. "Smt: 6 / 7 / 8") wins over the heading;
            # multi-value pools ("5/6/7/8") stay None instead of guessing a semester.
            if not course.pop('sem_explicit', False):
                course['semester'] = course.get('semester') or semester
            course['prodi'] = block.prodi
            course['kurikulum_tahun'] = KURIKULUM_MAP.get(block.prodi)
            course['sumber'] = f"{block.source_file}:L{row.line_num}"
            key = (block.prodi, course['kode_mk'])
            existing = courses.get(key)
            if existing is None:
                courses[key] = course
                order.append(key)
            else:
                def score(c: Dict) -> tuple:
                    return (c.get('semester') is None, not c.get('sks'))
                if score(course) < score(existing):
                    courses[key] = course
    rows_out = [courses[k] for k in order]
    for r in rows_out:
        r['id'] = f"{r['prodi']}-{r['kode_mk']}"
    if rows_out:
        fieldnames = ['id', 'prodi', 'kode_mk', 'nama_mk', 'semester', 'sks', 'sks_teori',
                      'sks_praktikum', 'sks_praktek', 'prasyarat', 'kurikulum_tahun', 'sumber']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"OK mata_kuliah.csv: {len(rows_out)} rows written")
    else:
        print("!! No course data found")

def build_semester_summary_csv(blocks: List[TableBlock], lines_by_file: Dict[str, List[str]], output_path: str):
    summaries = []
    for block in blocks:
        if not is_course_table(block):
            continue
        clean_leading_empty_column(block)
        block_idx = blocks.index(block)
        semester = extract_semester_from_context(blocks, block_idx,
                                                  lines_by_file.get(block.source_file, []))
        for row in block.rows:
            if is_totals_row(row.cells):
                summary = parse_totals_row(row.cells)
                if summary:
                    summary['prodi'] = block.prodi
                    summary['semester'] = semester
                    summary['sumber'] = f"{block.source_file}:L{row.line_num}"
                    summaries.append(summary)
                    break
    if summaries:
        seen = {}
        for s in summaries:
            key = f"{s['prodi']}-{s['semester'] if s['semester'] is not None else 'na'}"
            if key not in seen:
                s['id'] = key
                seen[key] = s
        summaries = list(seen.values())
        fieldnames = ['id', 'prodi', 'semester', 'sks_dokumen', 'sks_teori', 'sks_praktikum', 'sks_praktek', 'sumber']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        print(f"OK semester_summary.csv: {len(summaries)} rows written")
    else:
        print("!! No semester summaries found")

def build_enrichment_csv(blocks: List[TableBlock], output_path: str):
    enrichments = []
    for block in blocks:
        if not is_enrichment_table(block):
            continue
        clean_leading_empty_column(block)
        data_rows = [r for r in block.rows if not r.is_separator]
        if data_rows and not re.match(r'^\d+$', data_rows[0].cells[0].strip()):
            data_rows = data_rows[1:]
        for row in data_rows:
            cells = row.cells
            if len(cells) < 4:
                continue
            if not re.match(r'^\d+$', cells[0].strip()):
                continue
            kode_mk = cells[1].strip()
            if not re.match(r'^0?\d{6,8}$', kode_mk):
                continue
            enrichment = {'prodi': block.prodi, 'kode_mk': kode_mk,
                          'nama_mk': cells[2].strip(), 'sks': cells[3].strip(),
                          'sumber': f"{block.source_file}:L{row.line_num}"}
            enrichments.append(enrichment)
    if enrichments:
        seen = {}
        for e in enrichments:
            key = f"{e['prodi']}-{e['kode_mk']}"
            if key not in seen:
                e['id'] = key
                seen[key] = e
        enrichments = list(seen.values())
        fieldnames = ['id', 'prodi', 'kode_mk', 'nama_mk', 'sks', 'sumber']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enrichments)
        print(f"OK mata_kuliah_pengayaan.csv: {len(enrichments)} rows written")
    else:
        print("!! No enrichment data found")

def build_program_studi_csv(output_path: str):
    programs = [
        {'prodi': 'TI', 'nama_prodi': 'Teknik Informatika',
         'fakultas': 'Teknologi Industri dan Informatika', 'jenjang': 'S1', 'kurikulum_tahun': '2025-2026'},
        {'prodi': 'TRM', 'nama_prodi': 'Teknik Rekayasa Mekatronika',
         'fakultas': 'FTII', 'jenjang': 'D4', 'kurikulum_tahun': '2024'},
        {'prodi': 'TE', 'nama_prodi': 'Teknik Elektro',
         'fakultas': 'FTII', 'jenjang': 'S1', 'kurikulum_tahun': '2023-2024'},
        {'prodi': 'STI', 'nama_prodi': 'Sistem dan Teknologi Informasi',
         'fakultas': 'FTII', 'jenjang': 'S1', 'kurikulum_tahun': '2025-2026'},
        {'prodi': 'TM', 'nama_prodi': 'Teknik Mesin',
         'fakultas': 'FTII', 'jenjang': 'S1', 'kurikulum_tahun': '2026-2027'}
    ]
    fieldnames = ['prodi', 'nama_prodi', 'fakultas', 'jenjang', 'kurikulum_tahun']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(programs)
    print(f"OK program_studi.csv: {len(programs)} rows written (manual stub)")

def build_cpl_csv(output_path: str):
    print("!! cpl.csv: Not implemented (Priority 5)")

def build_cpl_mata_kuliah_csv(output_path: str):
    print("!! cpl_mata_kuliah.csv: Not implemented (Priority 6)")

def main():
    base_path = Path(__file__).parent.parent
    docs_dir = base_path / 'data' / 'documents'
    output_dir = base_path / 'data' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== UHAMKA Table Parser ===\n")
    md_files = list(docs_dir.glob('*.md'))
    print(f"Found {len(md_files)} markdown files\n")

    all_blocks = []
    lines_by_file = {}

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines_by_file[md_file.name] = lines
        print(f"Parsing {md_file.name}...")
        idx = 0
        file_blocks = []
        while idx < len(lines):
            if lines[idx].strip().startswith('|'):
                block = parse_markdown_table(lines, idx, str(md_file))
                if block:
                    file_blocks.append(block)
                    idx = block.end_line
                else:
                    idx += 1
            else:
                idx += 1
        print(f"  -> {len(file_blocks)} raw table blocks")
        all_blocks.extend(file_blocks)

    print(f"\nTotal raw blocks: {len(all_blocks)}")
    merged_blocks = merge_page_split_tables(all_blocks, lines_by_file)
    print(f"After merging page splits: {len(merged_blocks)} blocks\n")

    print("Building CSVs...\n")
    build_mata_kuliah_csv(merged_blocks, lines_by_file, str(output_dir / 'mata_kuliah.csv'))
    build_semester_summary_csv(merged_blocks, lines_by_file, str(output_dir / 'semester_summary.csv'))
    build_enrichment_csv(merged_blocks, str(output_dir / 'mata_kuliah_pengayaan.csv'))
    build_program_studi_csv(str(output_dir / 'program_studi.csv'))
    build_cpl_csv(str(output_dir / 'cpl.csv'))
    build_cpl_mata_kuliah_csv(str(output_dir / 'cpl_mata_kuliah.csv'))
    print("\nOK Done! CSVs written to data/tables/")

if __name__ == '__main__':
    main()
