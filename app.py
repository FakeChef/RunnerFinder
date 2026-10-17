"""
Runner Photo Finder - Streamlit interface.

Start with:   streamlit run app.py      (or double-click run_app.bat)

Phase 2: page layout, event selection, database connection.
Phase 7: the real Import / Search / Verify / Export functionality.
"""
from pathlib import Path
import csv

import streamlit as st

from config import get_config
from database.database import (get_or_create_event, init_db, list_events, reject_bib,
                               restore_bib)
from export.exporter import bib_export_folder
from search.photo_search import (filter_records, group_records, is_rejected,
                                 load_overrides, load_records, open_in_explorer,
                                 save_override, search_records)
from search.db_search import search as db_search

cfg = get_config()
st.set_page_config(page_title="Runner Photo Finder", page_icon="🏃", layout="wide")


@st.cache_resource
def get_db():
    """One shared database connection for the whole app session."""
    return init_db(cfg.paths.database_file)


conn = get_db()


def photo_for_display(path, photo_id=None) -> str:
    """Thumbnail if it exists, else a new thumbnail, else the original (V0.3.1).
    The original photo is only read; thumbnails live in data/thumbnails."""
    from scanner.photo_scanner import compute_fingerprint
    from utils.thumbnails import display_path
    fingerprint = None
    if photo_id is not None:
        row = conn.execute("SELECT fingerprint FROM photos WHERE id = ?", (photo_id,)).fetchone()
        fingerprint = row["fingerprint"] if row else None
    if fingerprint is None:
        try:          # CSV batches: same fingerprint as the scanner computes
            fingerprint = compute_fingerprint(path, Path(path).stat().st_size,
                                              cfg.scan.fingerprint_chunk_bytes)
        except OSError:
            return str(path)
    return str(display_path(path, cfg.paths.thumbnail_cache, fingerprint, cfg.thumbnails))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏃 Runner Photo Finder")
page = st.sidebar.radio("Page", ["Import photos", "Search", "Verify results", "Export"])

events = list_events(conn)
event_names = [e["name"] for e in events]
event_name = st.sidebar.selectbox(
    "Event", event_names, index=0 if event_names else None,
    placeholder="Create an event on the Import page",
)
st.sidebar.caption(f"Database: {cfg.paths.database_file}")

def search_sqlite_page():
    """Search on the SQLite index. Same rules as the CSV search below."""
    all_events = list_events(conn)
    if not all_events:
        st.info("索引中还没有活动。请先导入一个诊断批次：\n\n"
                "`.\.venv\Scripts\python.exe index_cli.py import-csv \"data\debug\ocr_...\" --event \"活动名称\"`")
        return
    names = [e["name"] for e in all_events]
    options = ["全部活动", *names]
    choice = st.selectbox("活动", options,
                          index=options.index(event_name) if event_name in names else 0, key="db_event")
    event_id = None if choice == "全部活动" else next(e["id"] for e in all_events if e["name"] == choice)
    st.caption("数字部分精确匹配并保留前导零。拒绝/恢复只写入 SQLite 人工反馈，不修改 OCR 结果或原始照片。")
    use_filter = st.checkbox("启用候选过滤", value=False, key="db_use_filter")
    show_filtered = st.checkbox("显示被过滤候选", value=False, disabled=not use_filter, key="db_show_filtered")
    min_len = st.number_input("最小数字 bib 长度", min_value=1, max_value=20, value=1,
                              disabled=not use_filter, key="db_min_len")
    max_len = st.number_input("最大数字 bib 长度", min_value=1, max_value=20, value=8,
                              disabled=not use_filter, key="db_max_len")
    excluded_text = st.text_input("排除的数字值（逗号分隔）", placeholder="例如：2026,10",
                                  disabled=not use_filter, key="db_excluded")
    excluded = {x.strip() for x in excluded_text.split(",") if x.strip().isdigit()}
    visible_rejected = st.checkbox("显示已拒绝候选", value=False, key="db_show_rejected")
    show_noise = st.checkbox("显示噪声过滤候选（附过滤原因）", value=False, key="db_show_noise",
                             help="自动过滤的候选（如存包号、10KM、2026）默认隐藏；数据未删除。")
    query = st.text_input("数字 bib", placeholder="18866 或 00123", key="db_query")
    if query and not query.strip().isdigit():
        st.warning("请输入纯数字；默认不做模糊匹配。")
    groups, filtered = db_search(conn, query, event_id,
                                 int(min_len) if use_filter else None,
                                 int(max_len) if use_filter else None,
                                 excluded if use_filter else None,
                                 include_rejected=visible_rejected, include_noise=show_noise)
    noise_hidden = [c for c in filtered if c.noise_filtered]
    filtered = [c for c in filtered if not c.noise_filtered]
    st.write(f"匹配照片：{len(groups)}；候选记录：{sum(len(g.candidates) for g in groups)}")
    if noise_hidden:
        st.caption(f"噪声过滤隐藏的候选：{len(noise_hidden)}（未删除；勾选“显示噪声过滤候选”可查看）")
    if use_filter:
        st.caption(f"过滤候选记录：{len(filtered)}（数据库中的 OCR 结果未删除）")
    for i, group in enumerate(groups):
        resolved = group.resolved
        display = group.candidates if visible_rejected else group.visible
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                if resolved.path:
                    st.image(photo_for_display(resolved.path, group.photo_id), use_container_width=True)
                else:
                    st.warning(f"照片{resolved.status}：{resolved.detail}")
            with c2:
                st.write(f"**数字：** `{group.numeric_identifier}`　**完整 bib：** "
                         + ", ".join(f"`{c.bib}`" for c in display))
                st.write(f"活动：{group.event_name}　文件：`{group.photo_identity}`")
                st.write("OCR 证据：" + "；".join(
                    f"{c.bib} / conf {c.confidence if c.confidence is not None else '-'} / "
                    f"readings {c.readings or '-'} / regions {c.region_ids or '-'} / {c.verification}"
                    + (f" / ⚠ 噪声过滤：{c.filter_reasons or '分数不足'}" if c.noise_filtered else "")
                    for c in display))
                if resolved.status == "missing":
                    st.error("原图不存在，未尝试替换路径。")
                elif st.button("在资源管理器中打开", key=f"db_open_{i}"):
                    open_in_explorer(resolved.path)
                st.caption(f"当前状态：{len(group.visible)} 个候选，{len(group.rejected)} 个已拒绝")
                for j, cand in enumerate(group.candidates):
                    if cand.verification in ("corrected", "added"):
                        continue                     # manual entries are not rejected here
                    if cand.rejected:
                        if visible_rejected and st.button(f"恢复 {cand.bib}", key=f"db_restore_{i}_{j}"):
                            restore_bib(conn, cand.photo_id, cand.bib)
                            st.rerun()
                    elif st.button(f"拒绝 {cand.bib}", key=f"db_reject_{i}_{j}"):
                        reject_bib(conn, cand.photo_id, cand.bib)
                        st.rerun()
    if show_filtered and filtered:
        st.subheader("被过滤候选（仅显示，不参与结果）")
        for cand in filtered:
            st.write(f"{cand.bib} → {cand.numeric_identifier}；{cand.event_name}；{cand.file_path}")
    if noise_hidden:
        with st.expander(f"噪声过滤隐藏的候选（{len(noise_hidden)}）"):
            for cand in noise_hidden:
                st.write(f"{cand.bib} → {cand.numeric_identifier}；原因：{cand.filter_reasons or '分数不足'}；"
                         f"{cand.event_name}；{cand.file_path}")


# ---------------------------------------------------------------- pages
if page == "Import photos":
    st.header("Import photos")
    folder = st.text_input("Photo folder", placeholder=r"D:\RacePhotos\Race")
    default_name = Path(folder).name if folder else ""
    new_event = st.text_input("Event name", value=default_name)
    if st.button("Create event", disabled=not new_event):
        get_or_create_event(conn, new_event.strip())
        st.success(f"Event '{new_event}' is ready.")
        st.rerun()
    st.info("Folder scanning works from the command line in Phase 3:\n\n"
            "`.\\.venv\\Scripts\\python.exe index_cli.py scan \"D:\\RacePhotos\\...\" --event \"Event name\"`\n\n"
            "Scanning and AI analysis from this page are added in Phase 7.")

elif page == "Search":
    st.header("Search by bib number")
    source = st.radio("数据来源", ["SQLite 索引", "CSV 批次（旧）"], horizontal=True,
                      help="SQLite 索引：已导入或已分析的活动。CSV 批次：原有的单批次诊断搜索（保留作备用）。")
    if source == "SQLite 索引":
        search_sqlite_page()
    else:
        st.caption("仅搜索当前选定批次；数字部分精确匹配并保留前导零。")
        default_root = Path(__file__).resolve().parent / "data"
        discovered = sorted(default_root.glob("**/photo_bibs.csv"), key=lambda p: str(p).lower())
        choices = [str(p) for p in discovered]
        csv_value = st.text_input("photo_bibs.csv 路径", value=choices[0] if choices else "")
        csv_path = Path(csv_value) if csv_value else None
        if not csv_path or not csv_path.is_file():
            st.info("请输入一个实际存在的诊断批次 photo_bibs.csv。")
        else:
            try:
                records = load_records(csv_path)
                overrides = load_overrides(csv_path)
            except (OSError, csv.Error) as exc:
                st.error(f"无法读取批次：{exc}")
                records, overrides = [], {}
            photo_root_value = st.text_input("可选原始照片根目录", placeholder=r"D:\RacePhotos")
            photo_root = Path(photo_root_value).expanduser() if photo_root_value else None
            if photo_root and not photo_root.is_dir():
                st.warning("照片根目录不存在；不会自动扫描其他位置。")
                photo_root = None
            use_filter = st.checkbox("启用候选过滤", value=False)
            show_filtered = st.checkbox("显示被过滤候选", value=False, disabled=not use_filter)
            min_len = st.number_input("最小数字 bib 长度", min_value=1, max_value=20, value=1,
                                      disabled=not use_filter)
            max_len = st.number_input("最大数字 bib 长度", min_value=1, max_value=20, value=8,
                                      disabled=not use_filter)
            excluded_text = st.text_input("排除的数字值（逗号分隔）", placeholder="例如：2026,10",
                                          disabled=not use_filter)
            excluded = {x.strip() for x in excluded_text.split(",") if x.strip().isdigit()}
            visible_rejected = st.checkbox("显示已拒绝候选", value=False)
            query = st.text_input("数字 bib", placeholder="18866 或 00123")
            if query and not query.strip().isdigit():
                st.warning("请输入纯数字；默认不做模糊匹配。")
            results = search_records(records, query)
            kept, filtered = filter_records(
                results, int(min_len) if use_filter else None,
                int(max_len) if use_filter else None,
                excluded if use_filter else None,
            )
            groups = group_records(kept, csv_path, photo_root)
            st.write(f"匹配照片：{len(groups)}；原始候选记录：{len(kept)}")
            if use_filter:
                st.caption(f"过滤候选记录：{len(filtered)}（原始 OCR 数据未删除）")
            for i, group in enumerate(groups):
                rejected_records = [r for r in group.records
                                    if is_rejected(overrides, csv_path, group.photo_identity, r.bib)]
                visible_records = [r for r in group.records if r not in rejected_records]
                if not visible_rejected and not visible_records:
                    continue
                resolved = group.resolved
                display_records = group.records if visible_rejected else visible_records
                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        if resolved.path:
                            st.image(photo_for_display(resolved.path), use_container_width=True)
                        else:
                            st.warning(f"照片{resolved.status}：{resolved.detail}")
                    with c2:
                        st.write(f"**数字：** `{group.numeric_identifier}`　**完整 bib：** "
                                 + ", ".join(f"`{r.bib}`" for r in display_records))
                        st.write(f"文件：`{display_records[0].file_value}`")
                        st.write("OCR 证据：" + "；".join(
                            f"{r.bib} / conf {r.confidence or '-'} / readings {r.readings or '-'} / regions {r.region_ids or '-'}"
                            for r in display_records))
                        if resolved.status == "ambiguous":
                            st.error("路径不唯一，未猜测实际照片。请修正 CSV 中的完整路径。")
                        elif resolved.status == "missing":
                            st.error("原图不存在，未尝试替换路径。")
                        elif st.button("在资源管理器中打开", key=f"open_{i}"):
                            open_in_explorer(resolved.path)
                        st.caption(f"当前状态：{len(visible_records)} 个候选，{len(rejected_records)} 个已拒绝")
                        for j, record in enumerate(group.records):
                            rejected = record in rejected_records
                            if rejected and not visible_rejected:
                                continue
                            if rejected:
                                if st.button(f"恢复 {record.bib}", key=f"restore_{i}_{j}"):
                                    save_override(csv_path, group.photo_identity, record.bib, "restore")
                                    st.rerun()
                            elif st.button(f"拒绝 {record.bib}", key=f"reject_{i}_{j}"):
                                save_override(csv_path, group.photo_identity, record.bib, "reject")
                                st.rerun()
            if show_filtered and filtered:
                st.subheader("被过滤候选（仅显示，不参与结果）")
                for record in filtered:
                    st.write(f"{record.bib} → {record.numeric_identifier}；{record.file_value}")

elif page == "Verify results":
    st.header("Verify results")
    st.write("Mark AI readings as correct or incorrect, fix wrong numbers, "
             "or add bibs the AI missed. Your corrections are saved for "
             "improving the model later.")
    st.info("Verification is added in Phase 7.")

elif page == "Export":
    st.header("Export")
    if event_name:
        example = bib_export_folder(cfg.paths.default_export_root, event_name, "10258")
        st.write("Photos will be copied to:")
        st.code(str(example))
    else:
        st.write("Create an event on the Import page first.")
    st.info("Export is added in Phase 7.")
