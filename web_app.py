import streamlit as st
import requests
import json
import os
import time
import re
import fitz  # PyMuPDF
import pandas as pd
import html
from collections import Counter
from openai import OpenAI
from anthropic import Anthropic

# ==================== 1. 页面基本配置 & CSS 终极魔法 ====================
st.set_page_config(page_title="ADS 文献分析引擎", page_icon="🌌", layout="wide")

# 💡 强力 CSS 控制：修复文字溢出，缩小字体，整体向上平移
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }
        /* 缩小按钮内的字体，适度减少内边距，让文字乖乖呆在框内 */
        div[data-testid="stButton"] button {
            padding: 0.2rem 0.5rem !important;
        }
        div[data-testid="stButton"] button p {
            font-size: 13px !important;
            white-space: nowrap !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 2. 本地配置与状态管理 ====================
CONFIG_FILE = "app_config.json"
default_config = {
    "ADS_API_TOKEN": "", "MINIMAX_API_KEY": "", "DEEPSEEK_API_KEY": "",
    "DOWNLOAD_DIR": "D:/ADS_Papers",
    "FONT_SIZE": 14,
    "THEME_MODE": "light",
    "SYSTEM_PROMPT": """你是一个资深的天体物理学研究员。请阅读提供的论文文本，寻找“类太阳恒星”的定义，重点关注Introduction、Sample Selection或Methods部分。请提取具体参数并严格以JSON格式输出：\n{\n    \"研究对象\": \"简明扼要说明研究的天体\",\n    \"类太阳恒星的定义标准\": \"总结作者界定类太阳恒星的具体物理参数标准。\",\n    \"论文概括\": \"100字以内概括核心工作\",\n    \"创新点\": \"一两句话概括创新之处\"\n}"""
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                for k, v in default_config.items():
                    if k not in cfg: cfg[k] = v
                return cfg
        except: pass
    return default_config.copy()

def save_config(config_data):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

if "config" not in st.session_state:
    st.session_state.config = load_config()

if "papers" not in st.session_state: st.session_state.papers = []
if "selected_bibcodes" not in st.session_state: st.session_state.selected_bibcodes = set()
if "total_found" not in st.session_state: st.session_state.total_found = 0
if "stop_process" not in st.session_state: st.session_state.stop_process = False
if "select_all_toggle" not in st.session_state: st.session_state.select_all_toggle = False
if "current_page" not in st.session_state: st.session_state.current_page = 0
if "sort_selector" not in st.session_state: st.session_state.sort_selector = "🔥 引用量 (由高到低)"

cfg = st.session_state.config

# ==================== 3. 左侧系统边栏 ====================
with st.sidebar:
    st.header("⚙️ 系统全局配置")
    st.markdown("---")
    
    cfg["DOWNLOAD_DIR"] = st.text_input("📁 PDF 保存路径", value=cfg["DOWNLOAD_DIR"])
    
    # 💡 终极安全写法：优先从云端 Secrets 读，如果云端没有，就回退去读本地的 cfg
    cfg["ADS_API_TOKEN"] = st.text_input("🔑 ADS API 密钥", value=st.secrets.get("ADS_API_TOKEN", cfg.get("ADS_API_TOKEN", "")), type="password")
    cfg["MINIMAX_API_KEY"] = st.text_input("🧠 MiniMax Token", value=st.secrets.get("MINIMAX_API_KEY", cfg.get("MINIMAX_API_KEY", "")), type="password")
    cfg["DEEPSEEK_API_KEY"] = st.text_input("🧠 DeepSeek Token", value=st.secrets.get("DEEPSEEK_API_KEY", cfg.get("DEEPSEEK_API_KEY", "")), type="password")
    
    st.markdown("---")
    st.subheader("📝 AI 批量提炼指令 (用于CSV)")
    cfg["SYSTEM_PROMPT"] = st.text_area("System Prompt", value=cfg["SYSTEM_PROMPT"], height=200)
    
    if st.button("💾 保存配置到本地", use_container_width=True):
        save_config(cfg)
        st.success("配置已永久保存！")

# ==================== 4. 后台业务逻辑库 ====================

def sanitize_filename(name): return re.sub(r'[^\w\-_.]', '_', name)

def search_ads(query, rows, sort_method):
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {cfg['ADS_API_TOKEN']}"}
    params = {"q": query, "fl": "bibcode,title,author,year,pubdate,doi,pub,bibstem,links_data,abstract,citation_count", "rows": rows, "sort": sort_method}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.total_found = data.get('response', {}).get('numFound', 0)
            return data.get("response", {}).get("docs", [])
        else:
            st.error(f"ADS 检索失败 (HTTP {resp.status_code})。请检查网络或 Token。")
    except Exception as e:
        st.error(f"网络异常: {str(e)}")
    return []

def find_download_links(paper):
    links = paper.get("links_data", [])
    urls = []
    for l in links:
        if isinstance(l, str):
            try: l = json.loads(l)
            except: continue
        t, u = l.get("type", ""), l.get("url", "")
        if t == "pdf": urls.append(("官方版 PDF", u))
        elif t == "preprint" and "arxiv" in u.lower():
            a_id = re.search(r'(\d{4}\.\d{4,5})', u)
            urls.append(("arXiv 预印本", f"https://arxiv.org/pdf/{a_id.group(1)}" if a_id else u))
    return urls

def download_file(url, filepath):
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and len(resp.content) > 1000 and resp.content[:5] == b'%PDF-':
            with open(filepath, "wb") as f: f.write(resp.content)
            return True, f"{len(resp.content)//1024} KB"
        return False, "非 PDF 数据"
    except: return False, "连接超时"

def call_ai_api(ai_type, prompt, json_mode=True):
    try:
        if ai_type == "minimax" and cfg["MINIMAX_API_KEY"]:
            client = Anthropic(api_key=cfg["MINIMAX_API_KEY"], base_url="https://api.minimaxi.com/anthropic")
            msg = client.messages.create(model="MiniMax-M2.7", max_tokens=8000, temperature=0.1, messages=[{"role": "user", "content": prompt}])
            res = "".join([b.text for b in msg.content if b.type == "text"])
            return json.loads(res.replace("```json", "").replace("```", "").strip()) if json_mode else res
        elif ai_type == "deepseek" and cfg["DEEPSEEK_API_KEY"]:
            client = OpenAI(api_key=cfg["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
            kwargs = {"model": "deepseek-chat", "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            msg = client.chat.completions.create(**kwargs)
            res = msg.choices[0].message.content.strip()
            return json.loads(res.replace("```json", "").replace("```", "").strip()) if json_mode else res
    except Exception as e:
        return f"AI 调用异常: {str(e)}" if not json_mode else None
    return "API 密钥未配置" if not json_mode else None

def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text.split("References")[0] if "References" in text else text
    except: return ""

def clean_abs_text(raw_text):
    if not raw_text: return "原文献未在数据库中提供官方摘要内容。"
    t = html.unescape(raw_text)
    t = re.sub(r'<[^>]+>', '', t)
    for cmd in [r'\left(', r'\right)', r'\rm', r'\mathrm', r'\mathbf']: t = t.replace(cmd, '')
    for k, v in {r'\Delta':'Δ', r'\Omega':'Ω', r'\alpha':'α', r'\beta':'β', r'\odot':'⊙', r'\sim':'~', r'\times':'×'}.items(): t = t.replace(k, v)
    return t.replace('$', '').replace('{', '').replace('}', '')

def step1_download_papers(papers, progress_callback=None):
    target_dir = cfg["DOWNLOAD_DIR"]
    os.makedirs(target_dir, exist_ok=True)
    meta_file = os.path.join(target_dir, "ads_metadata.json")
    meta_dict = json.load(open(meta_file, 'r', encoding='utf-8')) if os.path.exists(meta_file) else {}
        
    dl, pr = 0, 0
    total_pool = len(papers)
    
    for p in papers:
        if st.session_state.stop_process: break
        pr += 1
        if progress_callback: progress_callback(pr, total_pool)
        
        bib = p.get("bibcode", "unknown")
        fn = f"{sanitize_filename(bib)}.pdf"
        meta_dict[fn] = {
            "bibcode": bib, "标题": p.get("title", [""])[0], 
            "DOI": (p.get("doi", []) + ["无 DOI"])[0] if p.get("doi") else "无 DOI",
            "发表期刊": p.get("pub", "未知"), "年份": p.get("year", "未知"),
            "abstract": p.get("abstract", ""), "citation_count": p.get("citation_count", 0)
        }
        
        fp = os.path.join(target_dir, fn)
        if os.path.exists(fp): continue
        
        urls = find_download_links(p)
        success = False
        for name, url in urls:
            if download_file(url, fp)[0]:
                dl += 1
                success = True
                break
        time.sleep(1)
        
    with open(meta_file, 'w', encoding='utf-8') as f: json.dump(meta_dict, f, ensure_ascii=False, indent=4)
    return dl

def step2_extract_papers(papers, active_ai, progress_callback=None):
    target_dir = cfg["DOWNLOAD_DIR"]
    meta_file = os.path.join(target_dir, "ads_metadata.json")
    meta_dict = json.load(open(meta_file, 'r', encoding='utf-8')) if os.path.exists(meta_file) else {}
        
    output_csv = os.path.join(target_dir, f"Dataset_Extraction.csv")
    existing_files = pd.read_csv(output_csv)['文件名'].tolist() if os.path.exists(output_csv) else []
    results = pd.read_csv(output_csv).to_dict('records') if os.path.exists(output_csv) else []
        
    new_ext, pr, tot = 0, 0, len(papers)
    sys_prompt = cfg.get("SYSTEM_PROMPT")
    
    for p in papers:
        if st.session_state.stop_process: break
        pr += 1
        if progress_callback: progress_callback(pr, tot)
        
        bib = p.get("bibcode", "unknown")
        fn = f"{sanitize_filename(bib)}.pdf"
        fp = os.path.join(target_dir, fn)
        
        if not os.path.exists(fp) or fn in existing_files: continue
        
        txt = extract_text_from_pdf(fp)
        if not txt.strip(): continue
        
        extracted = call_ai_api(active_ai, f"{sys_prompt}\n\n【论文原始文本】\n{txt}", json_mode=True)
        if extracted and isinstance(extracted, dict):
            info = meta_dict.get(fn, {})
            results.append({"文件名": fn, "发表年份": p.get("year", ""), "文献标题": p.get("title", [""])[0], "发表期刊": p.get("pub", ""), **extracted})
            new_ext += 1
        
    if new_ext > 0: pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
    return new_ext

# ==================== 5. Web 弹窗 (Dialogs) ====================

@st.dialog("📄 文献详情与摘要", width="large")
def show_abstract_dialog(p, ai_type):
    title = p.get('title',['无标题'])[0]
    st.subheader(title)
    
    with st.container(border=True):
        authors = p.get("author", ["未知作者"])
        author_str = "; ".join(authors[:8]) + (f" 等 (共 {len(authors)} 人)" if len(authors) > 8 else "")
        st.markdown(f"**👤【作 者】**： {author_str}")
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**📚【期 刊】**： {p.get('pub', '未知期刊')}")
        c2.markdown(f"**🕒【时 间】**： {p.get('pubdate', '未知')}")
        
        c3, c4 = st.columns(2)
        c3.markdown("**🔗【DOI】**：")
        doi_val = p.get('doi', ['无'])[0] if p.get('doi') else "无"
        c3.code(doi_val, language=None)
        
        c4.markdown("**🔖【Bibcode】**：")
        c4.code(p.get('bibcode', '未知'), language=None)

    clean_text = clean_abs_text(p.get("abstract", ""))
    st.markdown("### 📝 摘要正文")
    st.write(clean_text)
    
    st.divider()
    if st.button("✨ 一键 AI 学术翻译 (简体中文)", type="primary"):
        with st.spinner("AI 正在运用专业天文学术语进行翻译，请稍候..."):
            prompt = (
                "Translate the following astrophysics abstract into standard Mainland Simplified Chinese (zh-CN). "
                "CRITICAL REQUIREMENT: You MUST use Simplified Chinese characters ONLY. "
                "Any use of Traditional Chinese characters (such as 體, 網, 導) is strictly prohibited. "
                "Output the translation directly without any prefix or suffix.\n\n"
                f"Abstract:\n{clean_text}"
            )
            res = call_ai_api(ai_type, prompt, json_mode=False)
            st.markdown("### 💡 AI 简体中文翻译")
            st.success(res)

@st.dialog("✨ AI 独立研读报告", width="large")
def show_ai_report_dialog(p, ai_type):
    title = p.get('title',['无标题'])[0]
    st.subheader(title)
    
    status_text = st.empty()
    report_area = st.empty()
    
    target_dir = cfg["DOWNLOAD_DIR"]
    os.makedirs(target_dir, exist_ok=True)
    bib = p.get("bibcode", "unknown")
    fp = os.path.join(target_dir, f"{sanitize_filename(bib)}.pdf")
    
    if not os.path.exists(fp):
        status_text.info("📥 本地无缓存，正在抓取论文 PDF...")
        urls = find_download_links(p)
        success = False
        for _, u in urls:
            if download_file(u, fp)[0]: 
                success = True; break
        if not success:
            status_text.error("❌ 获取 PDF 失败，可能被出版商拦截。")
            return
    
    status_text.info("📄 正在解析 PDF 文本...")
    txt = extract_text_from_pdf(fp)
    if not txt:
        status_text.error("❌ 文本提取失败 (可能为纯图片或加密 PDF)。")
        return
        
    status_text.info(f"🧠 {ai_type.upper()} 正在精读全文，请稍候 (约需10-20秒)...")
    prompt = f"你是资深天体物理研究员。请用中文给出全文总结：背景、方法、结果和结论。\n\n【文本】\n{txt}"
    res = call_ai_api(ai_type, prompt, json_mode=False)
    
    if res and "异常" not in res:
        status_text.success("✅ 报告生成完毕！")
        report_area.write(res)
        md_content = f"# {title}\n\n{res}"
        st.download_button("💾 保存报告为 Markdown", data=md_content, file_name=f"AI_Report_{sanitize_filename(title)}.md", mime="text/markdown", type="primary")
    else:
        status_text.error(f"❌ AI 通讯失败: {res}")


# ==================== 6. 主界面排版与检索 ====================

st.markdown("## 🌌 ADS 天体物理文献分析引擎")

is_expanded = len(st.session_state.papers) == 0

with st.expander("🔍 配置高级检索参数", expanded=is_expanded):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        keyword = st.text_input("目标关键词", "solar-type", help="支持逗号分隔多词 (AND)")
        start_date = st.text_input("检索起始期", "2022-01")
        author = st.text_input("指定作者", placeholder="留空则不限")
    with c2:
        loc = st.selectbox("检索限制域", ["全部字段 (All)", "仅标题 (Title)", "仅摘要 (Abstract)", "仅关键词 (Keyword)"])
        end_date = st.text_input("检索终止期", "2024-05")
        min_cite = st.number_input("最低引用量过滤", min_value=0, value=0)
    with c3:
        paper_type = st.text_input("目标期刊库", "ApJ, AJ, MNRAS, A&A", help="支持逗号分隔。留空为全库")
        max_rows = st.number_input("抓取上限量", min_value=1, max_value=2000, value=500)
    with c4:
        active_ai = st.selectbox("底层 AI 引擎", ["minimax", "deepseek"])
        do_ai_extract = st.checkbox("☑️ 下载后自动 AI 提炼 (CSV)", value=False)

loc_map = {"全部字段 (All)": "all", "仅标题 (Title)": "title", "仅摘要 (Abstract)": "abs", "仅关键词 (Keyword)": "keyword"}
sort_map = {
    "🔥 引用量 (由高到低)": "citation_count desc", "🔥 引用量 (由低到高)": "citation_count asc",
    "🕒 发表时间 (由新到旧)": "date desc", "🕒 发表时间 (由旧到新)": "date asc"
}

st.markdown("### 🚀 执行工作流")
col_act1, col_act2, col_act3, col_act4 = st.columns([2.5, 2, 3, 1.5])

with col_act1:
    if st.button("🔍 1. 检索网络文献", type="primary", use_container_width=True):
        st.session_state.selected_bibcodes.clear()
        st.session_state.current_page = 0
        st.session_state.select_all_toggle = False
        
        api_sort_method = sort_map[st.session_state.sort_selector]
        
        kw_str = keyword.strip()
        loc_val = loc_map[loc]
        if "," in kw_str or "，" in kw_str:
            kws = [k.strip() for k in re.split(r'[,，]', kw_str) if k.strip()]
            q_kw = " AND ".join([f'(title:"{k}" OR abs:"{k}" OR keyword:"{k}")' if loc_val == "all" else f'{loc_val}:"{k}"' for k in kws])
        else:
            q_kw = f'(title:"{kw_str}" OR abs:"{kw_str}" OR keyword:"{kw_str}")' if loc_val == "all" else f'{loc_val}:"{kw_str}"'
        
        q_str = f"({q_kw})"
        pt = paper_type.strip()
        if pt and pt.lower() != "all" and "全库" not in pt:
            j_parts = [f'bibstem:"{j.strip()}"' for j in re.split(r'[,，]', pt) if j.strip()]
            if j_parts: q_str += f' AND ({" OR ".join(j_parts)})'

        if start_date and end_date: q_str += f' AND pubdate:[{start_date} TO {end_date}]'
        if author.strip(): q_str += f' AND author:"{author.strip()}"'
        if min_cite > 0: q_str += f' AND citation_count:[{min_cite} TO *]'
        
        with st.spinner("正在穿越星际网络寻找论文..."):
            papers = search_ads(q_str, max_rows, api_sort_method)
            st.session_state.papers = papers
        st.rerun()

with col_act2:
    btn_label = "✅ 取消全选" if st.session_state.select_all_toggle else "⬜ 一键全选"
    if st.button(btn_label, use_container_width=True):
        st.session_state.select_all_toggle = not st.session_state.select_all_toggle
        if st.session_state.select_all_toggle:
            st.session_state.selected_bibcodes = set([p['bibcode'] for p in st.session_state.papers])
        else:
            st.session_state.selected_bibcodes.clear()
        st.rerun()

with col_act3:
    if st.button("📥 2. 批量处理选中项", type="primary", use_container_width=True):
        if not st.session_state.selected_bibcodes:
            st.warning("请先在下方列表中勾选需要处理的文献！")
        else:
            st.session_state.stop_process = False
            papers_to_process = [p for p in st.session_state.papers if p["bibcode"] in st.session_state.selected_bibcodes]
            tot = len(papers_to_process)
            
            progress_bar = st.progress(0, text="批量处理进度...")
            def dl_cb(curr, total): progress_bar.progress(curr/total * (0.5 if do_ai_extract else 1.0), text=f"下载进度: {curr}/{total}")
            dl_count = step1_download_papers(papers_to_process, dl_cb)
            
            new_ext = 0
            if do_ai_extract and not st.session_state.stop_process:
                def ai_cb(curr, total): progress_bar.progress(0.5 + curr/total * 0.5, text=f"AI 解析进度: {curr}/{total}")
                new_ext = step2_extract_papers(papers_to_process, active_ai, ai_cb)
            
            if st.session_state.stop_process:
                st.warning("⛔ 处理已紧急中止！数据已安全封存。")
            else:
                msg = f"✅ 批量处理完成！成功下载 {dl_count} 篇 PDF。"
                if do_ai_extract: msg += f" AI 成功提炼提取 {new_ext} 篇数据至 CSV。"
                st.success(msg)

with col_act4:
    if st.button("🛑 中止", use_container_width=True):
        st.session_state.stop_process = True
        st.toast("已发送停止指令...")


# ==================== 7. 展示检索结果区 ====================
if st.session_state.papers:
    st.markdown("---")
    
    # 💡 核心修复 1：通过自定义 HTML 实现标题与下拉框的完美水平对齐
    head_col, sort_col, _ = st.columns([5.5, 3.5, 3])
    with head_col:
        st.markdown(f"<div style='font-size: 1.35rem; font-weight: 600; margin-top: 0.6rem;'>📚 检索结果 (命中: {st.session_state.total_found} 篇，勾选: {len(st.session_state.selected_bibcodes)} 篇)</div>", unsafe_allow_html=True)
    with sort_col:
        st.selectbox(
            "排序方式", 
            ["🔥 引用量 (由高到低)", "🔥 引用量 (由低到高)", "🕒 发表时间 (由新到旧)", "🕒 发表时间 (由旧到新)"],
            key="sort_selector",
            label_visibility="collapsed"
        )
        
    sort_method = sort_map[st.session_state.sort_selector]
    
    current_list = st.session_state.papers.copy()
    if sort_method == "citation_count desc": current_list.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    elif sort_method == "citation_count asc": current_list.sort(key=lambda x: x.get("citation_count", 0), reverse=False)
    elif sort_method == "date desc": current_list.sort(key=lambda x: str(x.get("pubdate", "0000-00")), reverse=True)
    elif sort_method == "date asc": current_list.sort(key=lambda x: str(x.get("pubdate", "0000-00")), reverse=False)

    PAPERS_PER_PAGE = 10
    total_pages = max(1, (len(current_list) + PAPERS_PER_PAGE - 1) // PAPERS_PER_PAGE)
    
    if st.session_state.current_page >= total_pages: st.session_state.current_page = max(0, total_pages - 1)
        
    start_idx = st.session_state.current_page * PAPERS_PER_PAGE
    end_idx = start_idx + PAPERS_PER_PAGE
    page_papers = current_list[start_idx:end_idx]

    def change_page(delta):
        st.session_state.current_page += delta
    
    # 💡 核心修复 2：取消翻页按钮的无脑拉伸，让它自然显示，绝不折行
    pc1, pc2, pc3, pc4 = st.columns([1, 2, 1, 8])
    with pc1: st.button("⬅️ 上一页", key="prev_top", disabled=(st.session_state.current_page == 0), on_click=change_page, args=(-1,))
    with pc2: st.markdown(f"<div style='text-align: center; margin-top: 5px; font-weight: bold;'>当前页: {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
    with pc3: st.button("下一页 ➡️", key="next_top", disabled=(st.session_state.current_page >= total_pages - 1), on_click=change_page, args=(1,))

    for i, p in enumerate(page_papers):
        global_idx = start_idx + i + 1
        bib = p["bibcode"]
        title = p.get('title', ['无标题'])[0]
        date = p.get('pubdate', '?')
        cite = p.get('citation_count', 0)
        
        with st.container(border=True):
            # 💡 核心修复 3：扩大右侧动作按钮的列宽比例，彻底消除拥挤感
            col_chk, col_info, col_btn1, col_btn2, col_btn3 = st.columns([0.4, 6.6, 1.0, 1.0, 1.0])
            
            with col_chk:
                chk_key = f"chk_{bib}"
                st.session_state[chk_key] = (bib in st.session_state.selected_bibcodes)
                
                def sync_check(b=bib, k=chk_key):
                    if st.session_state[k]: st.session_state.selected_bibcodes.add(b)
                    else: st.session_state.selected_bibcodes.discard(b)
                        
                st.checkbox("", key=chk_key, on_change=sync_check)
                
            with col_info:
                st.markdown(f"**{global_idx}. [{date}] [引:{cite}] {title}**")
                
            with col_btn1:
                if st.button("📄 摘要", key=f"btn_abs_{bib}", use_container_width=True):
                    show_abstract_dialog(p, active_ai)
            with col_btn2:
                if st.button("✨ 研读", type="primary", key=f"btn_ai_{bib}", use_container_width=True):
                    show_ai_report_dialog(p, active_ai)
            with col_btn3:
                if st.button("📥 下载", key=f"btn_dl_{bib}", use_container_width=True):
                    fp = os.path.join(cfg["DOWNLOAD_DIR"], f"{sanitize_filename(bib)}.pdf")
                    if os.path.exists(fp):
                        st.info("已存在本地！")
                    else:
                        urls = find_download_links(p)
                        succ = False
                        for _, u in urls:
                            if download_file(u, fp)[0]: succ = True; break
                        if succ: st.toast("✅ 下载成功！")
                        else: st.error("❌ 下载失败")
                        
    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2, bc3, bc4 = st.columns([1, 2, 1, 8])
    with bc1: st.button("⬅️ 上一页", key="prev_bot", disabled=(st.session_state.current_page == 0), on_click=change_page, args=(-1,))
    with bc2: st.markdown(f"<div style='text-align: center; margin-top: 5px; font-weight: bold;'>当前页: {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
    with bc3: st.button("下一页 ➡️", key="next_bot", disabled=(st.session_state.current_page >= total_pages - 1), on_click=change_page, args=(1,))