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
        /* 1. 整体页面平移 */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }
        /* 2. 强制按钮内文字不换行，并缩小左右边距防止溢出 */
        div[data-testid="stButton"] button {
            padding: 0rem 0.6rem !important;
            height: 2.6rem !important;
            min-height: 2.6rem !important;
        }
        div[data-testid="stButton"] button p {
            font-size: 13.5px !important;
            white-space: nowrap !important;
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

# 💡 新增：在这里提前定义弹窗函数，这样走到下面第 3 部分时就不会报错了
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
    # 💡 彻底抛弃手动输入路径，系统在后台自动建一个临时文件夹存放下载的 PDF
    cfg["DOWNLOAD_DIR"] = "ADS_Papers_Temp"
    os.makedirs(cfg["DOWNLOAD_DIR"], exist_ok=True)
    
    # 安全获取密钥函数（兼容本地与云端）
    def get_safe_token(key):
        try:
            return st.secrets[key]
        except Exception:
            return cfg.get(key, "")

    cfg["ADS_API_TOKEN"] = st.text_input("🔑 ADS API 密钥", value=get_safe_token("ADS_API_TOKEN"), type="password")
    cfg["MINIMAX_API_KEY"] = st.text_input("🧠 MiniMax Token", value=get_safe_token("MINIMAX_API_KEY"), type="password")
    cfg["DEEPSEEK_API_KEY"] = st.text_input("🧠 DeepSeek Token", value=get_safe_token("DEEPSEEK_API_KEY"), type="password")
    
    st.subheader("📝 AI 提炼指令")
    # 把侧边栏的输入框变成一个小预览框
    cfg["SYSTEM_PROMPT"] = st.text_area("System Prompt (预览)", value=cfg["SYSTEM_PROMPT"], height=100)
    
    # 💡 召唤全屏编辑器的按钮
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
@st.dialog("🪟 全屏编辑 AI 提炼指令", width="large")
def show_prompt_editor_dialog():
    st.info("💡 在这里您可以拥有更宽广的视野来精细调优您的 System Prompt。")
    new_prompt = st.text_area("系统指令 (System Prompt)", value=cfg["SYSTEM_PROMPT"], height=450, label_visibility="collapsed")
    
    if st.button("💾 保存并应用指令", type="primary", use_container_width=True):
        cfg["SYSTEM_PROMPT"] = new_prompt
        save_config(cfg)
        st.rerun()

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
        # 💡 使用 Markdown 的行内代码语法，让标题和内容保持在同一行
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


# ==================== 6. 主界面 ====================
# ==================== 6. 主界面 ====================
st.markdown("## 🌌 ADS 天体物理文献分析引擎")

# 💡 新增：系统使用指南（折叠面板）
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
    # 💡 核心修复 2：优化列布局，并在复选框上方增加隐形占位符对齐
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
        # （ZIP 勾选框已经被删除了，因为现在是强制打包模式）

loc_map = {"全部字段 (All)": "all", "仅标题 (Title)": "title", "仅摘要 (Abstract)": "abs", "仅关键词 (Keyword)": "keyword"}
sort_map = {
    "🔥 引用量 (由高到低)": "citation_count desc", "🔥 引用量 (由低到高)": "citation_count asc",
    "🕒 发表时间 (由新到旧)": "date desc", "🕒 发表时间 (由旧到新)": "date asc"
}

st.markdown("### 🚀 执行工作流")
act1, act2, act3, act4 = st.columns([2.5, 2, 3, 1.5])

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
    btn_label = "✅ 取消全选" if st.session_state.select_all_toggle else "⬜ 一键全选"
    if st.button(btn_label, use_container_width=True):
        st.session_state.select_all_toggle = not st.session_state.select_all_toggle
        if st.session_state.select_all_toggle:
            st.session_state.selected_bibcodes = set([p['bibcode'] for p in st.session_state.papers])
        else:
            st.session_state.selected_bibcodes.clear()
        st.rerun()

with act3:
    if st.button("📥 2. 批量处理选中项", type="primary", use_container_width=True):
        if not st.session_state.selected_bibcodes:
            st.warning("请先勾选需要处理的文献！")
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
                st.warning("⛔ 处理已中止！")
            else:
                # 💡 无论本地还是云端，直接把下好的 PDF 和提炼的 CSV 强制打包成 ZIP
                zip_path = os.path.join(cfg["DOWNLOAD_DIR"], "ADS_Selected_Papers.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for bib in st.session_state.selected_bibcodes:
                        fn = f"{sanitize_filename(bib)}.pdf"
                        fp = os.path.join(cfg["DOWNLOAD_DIR"], fn)
                        if os.path.exists(fp):
                            zf.write(fp, arcname=fn)
                    csv_path = os.path.join(cfg["DOWNLOAD_DIR"], "Dataset_Extraction.csv")
                    if do_ai_extract and os.path.exists(csv_path):
                        zf.write(csv_path, arcname="Dataset_Extraction.csv")
                
                # 标记压缩包准备完毕
                st.session_state.zip_ready = True

                msg = f"✅ 批量处理完成！下载 {dl_count} 篇 PDF。"
                if do_ai_extract: msg += f" AI 提炼提取 {new_ext} 篇数据。"
                st.success(msg)
                
                # 停顿1秒后刷新页面，让旁边的下载按钮亮起
                time.sleep(1)
                st.rerun()

with act4:
    # 💡 专属 ZIP 下载按钮（去掉了勾选框的判断逻辑，只要有包就给下）
    zip_path = os.path.join(cfg["DOWNLOAD_DIR"], "ADS_Selected_Papers.zip")
    if st.session_state.get("zip_ready") and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button("📦 下载 ZIP", data=f, file_name="ADS_Selected_Papers.zip", mime="application/zip", use_container_width=True, type="primary")
    else:
        st.button("📦 下载 ZIP", disabled=True, use_container_width=True)


# ==================== 7. 展示结果 ====================
if st.session_state.papers:
    st.markdown("---")
    
    head_col, sort_col, _ = st.columns([6, 3, 3])
    with head_col:
        # 💡 核心修复 1：把“已勾选数量”完美加回标题中
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
            c_chk, c_info, c_b1, c_b2, c_b3 = st.columns([0.4, 6.3, 1.1, 1.1, 1.1])
            
            chk_key = f"chk_{bib}"
            st.session_state[chk_key] = (bib in st.session_state.selected_bibcodes)
            def sync_check(b=bib, k=chk_key):
                if st.session_state[k]: st.session_state.selected_bibcodes.add(b)
                else: st.session_state.selected_bibcodes.discard(b)
            
            c_chk.checkbox("", key=chk_key, on_change=sync_check)
            c_info.markdown(f"**{start_idx+i+1}.** [{p.get('pubdate','?')}] **{p.get('title',[''])[0]}**")
            
            if c_b1.button("📄 摘要", key=f"ab_{i}", use_container_width=True): show_abstract_dialog(p, active_ai)
            if c_b2.button("AI 研读", key=f"ai_{i}", use_container_width=True, type="primary"): show_ai_report_dialog(p, active_ai)
            if c_b3.button("📥 下载", key=f"dl_{i}", use_container_width=True):
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
    bc1, bc2, bc3, bc4 = st.columns([1.5, 2.5, 1.5, 6.5])
    with bc1: st.button("⬅️ 上一页", key="prev_bot", disabled=st.session_state.current_page==0, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page-1), use_container_width=True)
    with bc2: st.markdown(f"<div style='text-align: center; margin-top: 5px; font-weight: bold;'>页码: {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
    with bc3: st.button("下一页 ➡️", key="next_bot", disabled=st.session_state.current_page>=total_pages-1, on_click=lambda: setattr(st.session_state, 'current_page', st.session_state.current_page+1), use_container_width=True)