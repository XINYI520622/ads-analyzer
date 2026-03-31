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
import zipfile

# ==================== 1. 页面基本配置 & CSS 终极修正 ====================
st.set_page_config(page_title="ADS 文献分析引擎", page_icon="🌌", layout="wide")

# 💡 强力 CSS 控制：解决顶部留白、标题对齐、按钮文字出图问题
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem !important; }
        /* 强制按钮文字自动换行并垂直居中 */
        div[data-testid="stButton"] button {
            height: auto !important;
            min-height: 2.6rem !important;
            padding: 0.4rem 0.6rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[data-testid="stButton"] button p {
            white-space: normal !important;
            word-break: break-word !important;
            line-height: 1.2 !important;
            font-size: 13px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 2. 本地配置与状态管理 ====================
CONFIG_FILE = "app_config.json"
default_config = {
    "ADS_API_TOKEN": "", "MINIMAX_API_KEY": "", "DEEPSEEK_API_KEY": "",
    "DOWNLOAD_DIR": "D:/ADS_Papers",
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

# 状态变量初始化
for key, val in {
    "papers": [], "selected_bibcodes": set(), "total_found": 0,
    "stop_process": False, "select_all_toggle": False, "current_page": 0,
    "sort_selector": "🔥 引用量 (由高到低)"
}.items():
    if key not in st.session_state: st.session_state[key] = val

cfg = st.session_state.config

# 提前定义弹窗函数
@st.dialog("🪟 全屏编辑 AI 提炼指令", width="large")
def show_prompt_editor_dialog():
    st.info("💡 在这里您可以拥有更宽广的视野来精细调优您的 System Prompt。")
    new_prompt = st.text_area("系统指令 (System Prompt)", value=cfg["SYSTEM_PROMPT"], height=450, label_visibility="collapsed")
    
    if st.button("💾 保存并应用指令", type="primary", use_container_width=True):
        cfg["SYSTEM_PROMPT"] = new_prompt
        save_config(cfg)
        st.rerun()

# ==================== 3. 左侧系统边栏 ====================
with st.sidebar:
    st.header("⚙️ 系统全局配置")
    cfg["DOWNLOAD_DIR"] = "ADS_Papers_Temp"
    os.makedirs(cfg["DOWNLOAD_DIR"], exist_ok=True)
    
    def get_safe_token(key):
        try:
            return st.secrets[key]
        except Exception:
            return cfg.get(key, "")

    cfg["ADS_API_TOKEN"] = st.text_input("🔑 ADS API 密钥", value=get_safe_token("ADS_API_TOKEN"), type="password")
    cfg["MINIMAX_API_KEY"] = st.text_input("🧠 MiniMax Token", value=get_safe_token("MINIMAX_API_KEY"), type="password")
    cfg["DEEPSEEK_API_KEY"] = st.text_input("🧠 DeepSeek Token", value=get_safe_token("DEEPSEEK_API_KEY"), type="password")
    
    st.subheader("📝 AI 提炼指令")
    cfg["SYSTEM_PROMPT"] = st.text_area("System Prompt (预览)", value=cfg["SYSTEM_PROMPT"], height=100)
    
    if st.button("🪟 放大全屏编辑", use_container_width=True):
        show_prompt_editor_dialog()
        
    if st.button("💾 保存所有配置", use_container_width=True, type="primary"):
        save_config(cfg)
        st.success("配置已安全保存！")

# ==================== 4. 后台逻辑 ====================

def sanitize_filename(name): return re.sub(r'[^\w\-_.]', '_', name)

def search_ads(query, rows, sort_method):
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {cfg['ADS_API_TOKEN']}"}
    params = {"q": query, "fl": "bibcode,title,author,year,pubdate,doi,pub,abstract,citation_count,links_data", "rows": rows, "sort": sort_method}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.total_found = data.get('response', {}).get('numFound', 0)
            return data.get("response", {}).get("docs", [])
    except: pass
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
    
    # 💡 强力修复 1：安全的 JSON 加载，防止文件损坏导致整体崩溃
    meta_dict = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
        except Exception:
            pass
        
    dl_success_count = 0
    pr = 0
    total_pool = len(papers)
    
    for p in papers:
        if st.session_state.stop_process: break
        pr += 1
        bib = p.get("bibcode", "unknown")
        fn = f"{sanitize_filename(bib)}.pdf"
        
        meta_dict[fn] = {
            "bibcode": bib, "标题": p.get("title", [""])[0], 
            "DOI": (p.get("doi", []) + ["无 DOI"])[0] if p.get("doi") else "无 DOI",
            "发表期刊": p.get("pub", "未知"), "年份": p.get("year", "未知"),
            "abstract": p.get("abstract", ""), "citation_count": p.get("citation_count", 0)
        }
        
        fp = os.path.join(target_dir, fn)
        if os.path.exists(fp):
            dl_success_count += 1
            if progress_callback: progress_callback(pr, total_pool, f"已存在: {bib}")
            continue
        
        urls = find_download_links(p)
        success = False
        for name, url in urls:
            is_ok, msg = download_file(url, fp)
            if is_ok:
                dl_success_count += 1
                success = True
                break
        
        if progress_callback:
            status_msg = f"✅ 成功: {bib}" if success else f"❌ 失败: {bib}"
            progress_callback(pr, total_pool, status_msg)
        time.sleep(0.5)
        
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=4, default=lambda o: list(o) if isinstance(o, set) else str(o))
    return dl_success_count

def step2_extract_papers(papers, active_ai, progress_callback=None):
    target_dir = cfg["DOWNLOAD_DIR"]
    meta_file = os.path.join(target_dir, "ads_metadata.json")
    
    # 💡 强力修复 2：同步加入安全的 JSON 加载
    meta_dict = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
        except Exception:
            pass
        
    output_csv = os.path.join(target_dir, f"Dataset_Extraction.csv")
    existing_files = pd.read_csv(output_csv)['文件名'].tolist() if os.path.exists(output_csv) else []
    results = pd.read_csv(output_csv).to_dict('records') if os.path.exists(output_csv) else []
        
    new_ext, pr, tot = 0, 0, len(papers)
    sys_prompt = cfg.get("SYSTEM_PROMPT")
    
    for p in papers:
        if st.session_state.stop_process: break
        pr += 1
        if progress_callback: progress_callback(pr, tot, f"研读: {p.get('bibcode', '未知')}")
        
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
        doi_val = p.get('doi', ['无'])[0] if p.get('doi') else "无"
        c3.markdown(f"**🔗【DOI】**： `{doi_val}`")
        c4.markdown(f"**🔖【Bibcode】**： `{p.get('bibcode', '未知')}`")

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

# 💡 强力修复 3：使用独立的回调函数执行移除，这样弹窗绝对不会自己关闭！
def remove_from_cart(b):
    st.session_state.selected_bibcodes.discard(b)

@st.dialog("🛒 待下载文献清单", width="large")
def show_cart_dialog():
    selected_bibs = list(st.session_state.selected_bibcodes)
    selected_papers = [p for p in st.session_state.papers if p["bibcode"] in selected_bibs]
    
    if not selected_papers:
        st.info("清单已空，请点击弹窗外任意位置或右上角 X 关闭。")
        return

    st.markdown(f"### 📦 当前勾选 **{len(selected_papers)}** 篇")
    
    for i, p in enumerate(selected_papers):
        bib = p["bibcode"]
        col_text, col_btn = st.columns([8.5, 1.5])
        col_text.write(f"{i+1}. {p.get('title',[''])[0][:80]}...")
        # 绑定 on_click 回调，不主动触发 rerun，完美实现内部刷新不退弹窗！
        col_btn.button("❌ 移除", key=f"cart_rm_{bib}", on_click=remove_from_cart, args=(bib,), use_container_width=True)

def clear_cache_after_download():
    st.session_state.selected_bibcodes.clear()
    st.session_state.zip_ready = False
    st.session_state.select_all_toggle = False

# ==================== 6. 主界面 ====================
st.markdown("## 🌌 ADS 天体物理文献分析引擎")

with st.expander("📖 系统使用指南 (新手必读)", expanded=False):
    st.markdown("""
    欢迎使用本专属科研工作站！本工具专为天文学文献检索与数据提取打造，请按照以下步骤开启高效科研：
    
    * **⚙️ 零：配置环境**
        首次使用，请在左侧边栏填写您的 `ADS API 密钥` 及 `AI 引擎 Token`。下方的“AI 提炼指令”可自由修改（默认用于提取物理参数等核心指标）。
    * **🔍 一：精准检索**
        在下方展开“检索参数”，输入目标。例如输入 `solar-type`，或者查找《太阳物理学导论》相关的概念如 `solar flares`、`coronal mass ejections` 等。按需限制期刊库（如 `ApJ, A&A`），点击 **[检索文献]**。
    * **✨ 二：单篇速览与研读**
        在检索结果中，点击 **[📄 摘要]** 可查看 DOI 等详情，并支持一键强力转换标准简体中文；点击 **[AI 研读]** 可让 AI 代跑全文，直接生成“背景-方法-结论”总结报告。
    * **📦 三：批量下载与数据提炼（云端终极连招）**
        1️⃣ 勾选所需的文献（或一键全选）。
        2️⃣ 点击红色的 **[📥 2. 批量处理选中项]**。云端服务器会开始狂飙，在后台自动抓取 PDF 并用 AI 提炼数据为 Excel(CSV) 表格。
        3️⃣ 等待提示成功后，点击旁边亮起的 **[📦 下载 ZIP]** 按钮，将所有战利品一键打包带回你的电脑！
    """)

with st.expander("🔍 检索参数", expanded=len(st.session_state.papers)==0):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        keyword = st.text_input("关键词", "solar-type")
        start_date = st.text_input("起始期", "2022-01")
    with c2:
        loc = st.selectbox("检索域", ["全部字段 (All)", "仅标题 (Title)", "仅摘要 (Abstract)", "仅关键词 (Keyword)"])
        end_date = st.text_input("终止期", "2024-05")
    with c3:
        paper_type = st.text_input("期刊库", "ApJ, AJ, MNRAS, A&A")
        max_rows = st.number_input("上限量", value=500)
    with c4:
        active_ai = st.selectbox("AI 引擎", ["minimax", "deepseek"])
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        do_ai_extract = st.checkbox("自动 AI 提炼 (CSV)")

loc_map = {"全部字段 (All)": "all", "仅标题 (Title)": "title", "仅摘要 (Abstract)": "abs", "仅关键词 (Keyword)": "keyword"}
sort_map = {
    "🔥 引用量 (由高到低)": "citation_count desc", "🔥 引用量 (由低到高)": "citation_count asc",
    "🕒 发表时间 (由新到旧)": "date desc", "🕒 发表时间 (由旧到新)": "date asc"
}

st.markdown("### 🚀 执行工作流")
act1, act2, act3, act4, act5 = st.columns([1.8, 1.6, 1.8, 2.8, 1.2])

with act1:
    if st.button("🔍 1. 检索文献", type="primary", use_container_width=True):
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
        
        with st.spinner("检索中..."):
            st.session_state.papers = search_ads(q_str, max_rows, api_sort_method)
        st.rerun()

with act2:
    if len(st.session_state.selected_bibcodes) > 0:
        if st.button("🗑️ 清空购物车", use_container_width=True):
            clear_cache_after_download()
            st.rerun()
    else:
        btn_disabled = len(st.session_state.papers) == 0
        if st.button("🛒 全部加购", use_container_width=True, disabled=btn_disabled):
            st.session_state.selected_bibcodes = set([p['bibcode'] for p in st.session_state.papers])
            st.rerun()

with act3:
    cart_count = len(st.session_state.selected_bibcodes)
    if cart_count > 0:
        if st.button(f"🛒 查看/修改清单 ({cart_count})", use_container_width=True):
            show_cart_dialog()
    else:
        st.button(f"🛒 查看清单 (0)", disabled=True, use_container_width=True)

with act4:
    if cart_count > 0:
        kw_clean = sanitize_filename(keyword.strip()) if keyword.strip() else "All"
        zip_name = f"ADS_{kw_clean}.zip"
        zip_path = os.path.join(cfg["DOWNLOAD_DIR"], zip_name)
        
        # 1. 未生成压缩包：显示处理进度流
        if not st.session_state.get("zip_ready"):
            if st.button("🚀 2. 一键打包并下载", type="primary", use_container_width=True):
                with st.status("🛠️ 正在处理，请稍后...", expanded=True) as status:
                    papers_to_process = [p for p in st.session_state.papers if p["bibcode"] in st.session_state.selected_bibcodes]
                    
                    p_bar = st.progress(0)
                    p_text = st.empty()
                    
                    def update_ui(curr, total, msg):
                        p_bar.progress(curr/total)
                        p_text.text(f"进度: {curr}/{total} - {msg}")

                    final_dl_count = step1_download_papers(papers_to_process, update_ui)
                    
                    # 💡 强力修复 4：修复了缺失的 ZIP 打包实际逻辑和缩进！
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for bib in st.session_state.selected_bibcodes:
                            fn = f"{sanitize_filename(bib)}.pdf"
                            fp = os.path.join(cfg["DOWNLOAD_DIR"], fn)
                            if os.path.exists(fp): 
                                zf.write(fp, arcname=fn)
                        csv_path = os.path.join(cfg["DOWNLOAD_DIR"], "Dataset_Extraction.csv")
                        if do_ai_extract and os.path.exists(csv_path): 
                            zf.write(csv_path, arcname="Dataset_Extraction.csv")
                    
                    st.session_state.zip_ready = True
                    st.session_state.current_zip_path = zip_path
                    status.update(label="✅ 打包完成！点击下方按钮保存", state="complete")
                    st.rerun()
        
        # 2. 已生成压缩包：变成红色下载按钮
        else:
            with open(st.session_state.current_zip_path, "rb") as f:
                st.download_button(
                    label="💾 3. 立即保存 ZIP 到本地",
                    data=f,
                    file_name=os.path.basename(st.session_state.current_zip_path),
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                    on_click=clear_cache_after_download
                )
    else:
        st.button("🚀 2. 处理下载", disabled=True, use_container_width=True)

with act5:
    if st.button("🛑 中止", use_container_width=True):
        st.session_state.stop_process = True


# ==================== 7. 展示结果 ====================
if st.session_state.papers:
    st.markdown("---")
    
    head_col, sort_col, _ = st.columns([6, 3, 3])
    with head_col:
        st.markdown(f"<div style='font-size: 1.35rem; font-weight: 600; margin-top: 0.7rem;'>📚 检索结果 (命中: {st.session_state.total_found} 篇，已勾选: {len(st.session_state.selected_bibcodes)} 篇)</div>", unsafe_allow_html=True)
    with sort_col:
        st.selectbox("排序", ["🔥 引用量 (由高到低)", "🔥 引用量 (由低到高)", "🕒 发表时间 (由新到旧)", "🕒 发表时间 (由旧到新)"], key="sort_selector", label_visibility="collapsed")
    
    sort_method = sort_map[st.session_state.sort_selector]
    current_list = st.session_state.papers.copy()
    if sort_method == "citation_count desc": current_list.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    elif sort_method == "citation_count asc": current_list.sort(key=lambda x: x.get("citation_count", 0), reverse=False)
    elif sort_method == "date desc": current_list.sort(key=lambda x: str(x.get("pubdate", "0000-00")), reverse=True)
    elif sort_method == "date asc": current_list.sort(key=lambda x: str(x.get("pubdate", "0000-00")), reverse=False)

    PAPERS_PER_PAGE = 10
    total_pages = max(1, (len(current_list) + PAPERS_PER_PAGE - 1) // PAPERS_PER_PAGE)
    if st.session_state.current_page >= total_pages: st.session_state.current_page = max(0, total_pages - 1)
    
    pc1, pc2, pc3, pc4 = st.columns([1.5, 2.5, 1.5, 6.5])
    with pc1: st.button("⬅️ 上一页", key="prev_top", disabled=st.session_state.current_page==0, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page-1), use_container_width=True)
    with pc2: st.markdown(f"<div style='text-align: center; margin-top: 5px; font-weight: bold;'>页码: {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
    with pc3: st.button("下一页 ➡️", key="next_top", disabled=st.session_state.current_page>=total_pages-1, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page+1), use_container_width=True)

    start_idx = st.session_state.current_page * PAPERS_PER_PAGE
    for i, p in enumerate(current_list[start_idx:start_idx+PAPERS_PER_PAGE]):
        with st.container(border=True):
            bib = p["bibcode"]
            
            c_info, c_b1, c_b2, c_b3 = st.columns([6.7, 1.1, 1.1, 1.1])
            c_info.markdown(f"**{start_idx+i+1}.** [{p.get('pubdate','?')}] **{p.get('title',[''])[0]}**")
            
            if c_b1.button("📄 摘要", key=f"ab_{i}", use_container_width=True): show_abstract_dialog(p, active_ai)
            if c_b2.button("AI 研读", key=f"ai_{i}", use_container_width=True, type="primary"): show_ai_report_dialog(p, active_ai)
            
            if bib in st.session_state.selected_bibcodes:
                if c_b3.button("❌ 移出清单", key=f"dl_{i}", use_container_width=True):
                    st.session_state.selected_bibcodes.discard(bib)
                    st.rerun() 
            else:
                if c_b3.button("➕ 加清单", key=f"dl_{i}", use_container_width=True):
                    st.session_state.selected_bibcodes.add(bib)
                    st.rerun() 
                    
    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2, bc3, bc4 = st.columns([1.5, 2.5, 1.5, 6.5])
    with bc1: st.button("⬅️ 上一页", key="prev_bot", disabled=st.session_state.current_page==0, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page-1), use_container_width=True)
    with bc2: st.markdown(f"<div style='text-align: center; margin-top: 5px; font-weight: bold;'>页码: {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
    with bc3: st.button("下一页 ➡️", key="next_bot", disabled=st.session_state.current_page>=total_pages-1, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page+1), use_container_width=True)