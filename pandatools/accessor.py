import json
import warnings
import hashlib
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"
_MAGENTA = "\033[35m"
_BLUE   = "\033[34m"

class SonConfig:
    model = None
    api_key = None
    provider = "openai"
    is_ready = False
    
    ENV_KEYS = [
        "OPENAI_API_KEY",
        "SON_AI_KEY",
        "AI_API_KEY",
    ]

class BigDataConfig:
    enabled = False
    chunk_size = 100_000
    sample_size = 10_000
    low_memory = False

TASKS: Dict[int, Dict] = {}

def son_ai(task_id: int, prompt: str, desc: str = ""):
    def wrap(fn):
        TASKS[task_id] = {
            "id": task_id, 
            "prompt": prompt, 
            "desc": desc or prompt[:50], 
            "fn": fn
        }
        return fn
    return wrap

@son_ai(1, "Xóa tất cả hàng trùng lặp (duplicate rows) dựa trên tất cả columns", "🗑️  Xóa duplicate rows")
def _t1(df): 
    return df.drop_duplicates()

@son_ai(2, "Chuẩn hóa tên cột: lowercase, thay space và dấu gạch bằng underscore, bỏ ký tự đặc biệt, giới hạn 50 ký tự", "📝 Chuẩn hóa tên cột")
def _t2(df):
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_')[:50] for c in df.columns]
    return df

@son_ai(3, "Strip khoảng trắng thừa đầu/cuối tất cả string columns. Chuỗi rỗng, 'nan', 'None', 'null' thành NaN", "✂️  Strip text columns")
def _t3(df):
    for c in df.select_dtypes(include=['object', 'string']).columns:
        df[c] = df[c].astype(str).str.strip().replace(['', 'nan', 'None', 'null', 'NULL'], None)
    return df

@son_ai(4, "Xóa cột có hơn 70% missing values hoặc cột chỉ có 1 giá trị duy nhất (constant)", "🚫 Xóa cột vô nghĩa")
def _t4(df):
    to_drop = [c for c in df.columns if df[c].isna().mean() > 0.7 or df[c].nunique() <= 1]
    return df.drop(columns=to_drop) if to_drop else df

@son_ai(5, "Tự động sửa dtype: object chứa số→numeric, object chứa ngày→datetime, object ít unique→category, float toàn int→Int64, int 0/1→bool", "🔧 Auto-fix dtypes")
def _t5(df):
    for c in df.select_dtypes(include=['object']).columns:
        try:
            n = pd.to_numeric(df[c], errors='coerce')
            if n.notna().mean() > 0.9: 
                df[c] = n
                continue
        except: pass
        try:
            d = pd.to_datetime(df[c], errors='coerce', format='mixed')
            if d.notna().mean() > 0.9: 
                df[c] = d
                continue
        except: pass
        if df[c].nunique() <= 20: 
            df[c] = df[c].astype('category')
    
    for c in df.select_dtypes(include=['float']).columns:
        non_null = df[c].dropna()
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            df[c] = df[c].astype('Int64')
    
    for c in df.select_dtypes(include=['int']).columns:
        if set(df[c].dropna().unique()).issubset({0, 1}):
            df[c] = df[c].astype(bool)
    
    return df

@son_ai(6, "Fill missing values: numeric→median, datetime→forward fill, string/category→mode hoặc 'Unknown'", "💉 Fill missing values")
def _t6(df):
    for c in df.columns:
        if df[c].isna().sum() == 0: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].ffill()
        else:
            mode_val = df[c].mode()
            df[c] = df[c].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
    return df

@son_ai(7, "Phát hiện và xử lý outliers bằng IQR: tính Q1, Q3, IQR, clip ngoài [Q1-1.5*IQR, Q3+1.5*IQR] cho tất cả numeric columns", "📊 Xử lý outliers (IQR)")
def _t7(df):
    for c in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
    return df

@son_ai(8, "Chuẩn hóa text Unicode NFC (tốt cho tiếng Việt), thay multiple spaces bằng single space, strip whitespace", "🔤 Chuẩn hóa text Unicode")
def _t8(df):
    import unicodedata
    for c in df.select_dtypes(include=['object', 'string']).columns:
        df[c] = df[c].astype(str).apply(
            lambda x: unicodedata.normalize('NFC', x) if pd.notna(x) else x
        )
        df[c] = df[c].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

@son_ai(9, "Tạo features mới từ datetime columns: year, month, day, dayofweek, quarter, is_weekend", "📅 Feature engineering (datetime)")
def _t9(df):
    for c in df.select_dtypes(include=['datetime64']).columns:
        df[f'{c}_year'] = df[c].dt.year
        df[f'{c}_month'] = df[c].dt.month
        df[f'{c}_day'] = df[c].dt.day
        df[f'{c}_dayofweek'] = df[c].dt.dayofweek
        df[f'{c}_quarter'] = df[c].dt.quarter
        df[f'{c}_is_weekend'] = df[c].dt.dayofweek.isin([5,6]).astype(int)
    return df

@son_ai(10, "Tối ưu memory: downcast int64→int nhỏ hơn, float64→float32 khi có thể", "💾 Memory optimization")
def _t10(df):
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    return df

def _analyze(col: pd.Series) -> List[Dict[str, str]]:
    name   = col.name
    dtype  = col.dtype
    n      = len(col)
    n_miss = int(col.isna().sum())
    pct    = n_miss / n if n > 0 else 0
    items: List[Dict[str, str]] = []

    if pct > 0.6:
        items.append({
            "issue": f"Quá nhiều missing ({n_miss}/{n} = {pct*100:.1f}%)",
            "suggestion": "Cân nhắc DROP cột này",
            "code": f"df = df.drop(columns=['{name}'])",
        })
    elif n_miss > 0:
        if pd.api.types.is_numeric_dtype(dtype):
            items.append({
                "issue": f"Missing {n_miss} giá trị số",
                "suggestion": "Fill bằng median (hoặc mean)",
                "code": f"df['{name}'] = df['{name}'].fillna(df['{name}'].median())",
            })
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            items.append({
                "issue": f"Missing {n_miss} giá trị datetime",
                "suggestion": "Forward-fill theo thời gian",
                "code": f"df['{name}'] = df['{name}'].ffill()",
            })
        else:
            items.append({
                "issue": f"Missing {n_miss} giá trị chuỗi",
                "suggestion": "Fill bằng nhãn mặc định",
                "code": f"df['{name}'] = df['{name}'].fillna('Unknown')",
            })

    if dtype == object:
        try:
            pd.to_numeric(col.dropna())
            items.append({
                "issue": "Cột số đang lưu dạng object/str",
                "suggestion": "Chuyển sang kiểu số",
                "code": f"df['{name}'] = pd.to_numeric(df['{name}'], errors='coerce')",
            })
        except (ValueError, TypeError):
            pass

        sample = col.dropna().head(20).astype(str)
        date_like = sample.str.match(
            r"^\d{4}[-/]\d{2}[-/]\d{2}|^\d{2}[-/]\d{2}[-/]\d{4}"
        ).mean()
        if date_like > 0.5:
            items.append({
                "issue": "Cột ngày đang lưu dạng str",
                "suggestion": "Chuyển sang datetime",
                "code": f"df['{name}'] = pd.to_datetime(df['{name}'], errors='coerce')",
            })

        if col.dropna().astype(str).str.contains(r"^\s|\s$").any():
            items.append({
                "issue": "Có khoảng trắng thừa ở đầu/cuối",
                "suggestion": "Strip whitespace",
                "code": f"df['{name}'] = df['{name}'].str.strip()",
            })

        n_unique = col.nunique()
        if n > 0 and 0 < n_unique <= 50 and n_unique / n < 0.05:
            items.append({
                "issue": f"Chỉ có {n_unique} giá trị duy nhất (object tốn bộ nhớ)",
                "suggestion": "Chuyển sang category để tiết kiệm RAM",
                "code": f"df['{name}'] = df['{name}'].astype('category')",
            })

    elif pd.api.types.is_float_dtype(dtype):
        non_null = col.dropna()
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            items.append({
                "issue": "Cột float nhưng toàn số nguyên",
                "suggestion": "Chuyển sang int (hoặc Int64 nếu có NaN)",
                "code": f"df['{name}'] = df['{name}'].astype('Int64')",
            })

    elif pd.api.types.is_integer_dtype(dtype):
        if set(col.dropna().unique()).issubset({0, 1}):
            items.append({
                "issue": "Cột int chỉ chứa 0 và 1",
                "suggestion": "Chuyển sang bool cho rõ nghĩa",
                "code": f"df['{name}'] = df['{name}'].astype(bool)",
            })

    if n > 0 and col.nunique() <= 1:
        items.append({
            "issue": "Cột chỉ có 1 giá trị duy nhất (Constant)",
            "suggestion": "Nên DROP vì không mang lại thông tin",
            "code": f"df = df.drop(columns=['{name}'])",
        })

    if col.name and str(col.name).lower() in {"id", "key", "index", "uuid"}:
        n_dup = int(col.duplicated().sum())
        if n_dup > 0:
            items.append({
                "issue": f"{n_dup} giá trị ID bị trùng",
                "suggestion": "Kiểm tra và loại bỏ duplicate",
                "code": f"df = df.drop_duplicates(subset=['{name}'])",
            })

    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        non_null = col.dropna()
        if len(non_null) > 10:
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((non_null < q1 - 1.5*iqr) | (non_null > q3 + 1.5*iqr)).sum())
            if n_out > 0:
                items.append({
                    "issue": f"{n_out} outlier tiềm năng (ngoài IQR ×1.5)",
                    "suggestion": "Clip theo IQR hoặc xem xét thủ công",
                    "code": (
                        f"q1, q3 = df['{name}'].quantile([0.25, 0.75])\n"
                        f"iqr = q3 - q1\n"
                        f"df['{name}'] = df['{name}'].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)"
                    ),
                })

    return items


def _flat_suggest(items: List[Dict]) -> str:
    if not items:
        return "✅ OK"
    return " | ".join(it["suggestion"] for it in items)


def _flat_code(items: List[Dict]) -> str:
    if not items:
        return ""
    return " ; ".join(it["code"].replace("\n", " ") for it in items)


def _get_api_key(key=None):
    if isinstance(key, int):
        if 1 <= key <= len(SonConfig.ENV_KEYS):
            env_key = os.environ.get(SonConfig.ENV_KEYS[key-1], "")
            if env_key:
                return env_key
    
    if isinstance(key, str) and key:
        return key
    
    for env_key in SonConfig.ENV_KEYS:
        k = os.environ.get(env_key, "")
        if k:
            return k
    
    return None


def _ai_code(prompt: str, df: pd.DataFrame) -> str:
    
    if not SonConfig.is_ready:
        print(f"{_RED}❌ SƠN AI chưa khởi tạo! Dùng: df.clean.ai_son(model='gpt-4o', key='...'){_RESET}")
        return None
    
    h = hashlib.md5(f"{SonConfig.model}{prompt}{list(df.columns)}".encode()).hexdigest()[:8]
    
    cache_dir = os.path.expanduser("~/.son_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{h}.py")
    
    if os.path.exists(cache_file):
        print(f"      📦 Cache")
        return open(cache_file).read()
    
    try:
        if SonConfig.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=SonConfig.api_key)
            
            ctx = f"Columns: {list(df.columns)}\nDtypes: {df.dtypes.to_dict()}\nShape: {df.shape}\nSample:\n{df.head(3)}"
            
            r = client.chat.completions.create(
                model=SonConfig.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Viết hàm Python def clean(df): xử lý DataFrame. CHỈ code, KHÔNG markdown, KHÔNG giải thích. Return df."
                    },
                    {
                        "role": "user", 
                        "content": f"Yêu cầu: {prompt}\n\nDataFrame:\n{ctx}\n\ndef clean(df):"
                    }
                ],
                temperature=0.1
            )
            
            code = r.choices[0].message.content
            code = code.replace("```python", "").replace("```", "").strip()
            
        elif SonConfig.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=SonConfig.api_key)
            model = genai.GenerativeModel(SonConfig.model)
            
            ctx = f"Columns: {list(df.columns)}\nDtypes: {df.dtypes.to_dict()}\nShape: {df.shape}\nSample:\n{df.head(3)}"
            
            full_prompt = f"""Viết hàm Python def clean(df): xử lý DataFrame. CHỈ code, KHÔNG markdown, KHÔNG giải thích. Return df.

Yêu cầu: {prompt}

DataFrame:
{ctx}

def clean(df):"""
            
            response = model.generate_content(full_prompt)
            code = response.text.strip()
            code = code.replace("```python", "").replace("```", "").strip()
        
        else:
            print(f"{_RED}❌ Provider '{SonConfig.provider}' không hỗ trợ{_RESET}")
            return None
        
        open(cache_file, 'w').write(code)
        return code
        
    except ImportError as e:
        print(f"{_RED}❌ Thiếu thư viện: {e}{_RESET}")
        return None
    except Exception as e:
        print(f"{_RED}❌ API lỗi: {e}{_RESET}")
        return None


def son(df: pd.DataFrame, tasks: List[int], verbose: bool = True) -> pd.DataFrame:
    
    if not SonConfig.is_ready:
        print(f"{_RED}❌ SƠN AI chưa khởi tạo!{_RESET}")
        print(f"{_YELLOW}👉 Dùng: df.clean.ai_son(model='gpt-4o', key='...'){_RESET}")
        return df
    
    result = df.copy()
    
    if verbose:
        print(f"\n{_BOLD}{_BLUE}{'='*60}{_RESET}")
        print(f"{_BOLD}{_BLUE}🧠 SƠN AI | Model: {SonConfig.model} | Tasks: {tasks}{_RESET}")
        print(f"{_BOLD}{_BLUE}{'='*60}{_RESET}\n")
    
    for i, tid in enumerate(tasks, 1):
        if tid not in TASKS:
            if verbose:
                print(f"{_RED}❌ Task {tid} không tồn tại{_RESET}")
            continue
        
        task = TASKS[tid]
        
        if verbose:
            print(f"{_CYAN}[{i}/{len(tasks)}]{_RESET} {_BOLD}Task {tid}: {task['desc']}{_RESET}")
        
        code = _ai_code(task['prompt'], result)
        
        if code:
            try:
                env = {"df": result.copy(), "pd": pd, "np": np}
                exec(code, env)
                
                if "clean" in env:
                    result = env["clean"](result.copy())
                elif "df" in env:
                    result = env["df"]
                
                if verbose:
                    print(f"   {_GREEN}✅ {df.shape} → {result.shape}{_RESET}\n")
                    
            except Exception as e:
                if verbose:
                    print(f"   {_RED}❌ Lỗi: {e}{_RESET}")
                    print(f"   {_YELLOW}⤷ Fallback...{_RESET}")
                result = task['fn'](result.copy())
                if verbose:
                    print(f"   {_GREEN}✅ {result.shape}{_RESET}\n")
        else:
            result = task['fn'](result.copy())
            if verbose:
                print(f"   {_YELLOW}📌 Fallback: {result.shape}{_RESET}\n")
    
    if verbose:
        print(f"{_BOLD}{_GREEN}{'='*60}{_RESET}")
        print(f"{_BOLD}{_GREEN}✅ HOÀN TẤT: {df.shape} → {result.shape}{_RESET}")
        print(f"{_BOLD}{_GREEN}{'='*60}{_RESET}\n")
    
    return result


def list_tasks():
    print(f"\n{_BOLD}{_BLUE}📋 SƠN AI TASKS{_RESET}")
    print(f"{_BOLD}{_BLUE}{'='*60}{_RESET}\n")
    
    for tid in sorted(TASKS):
        t = TASKS[tid]
        print(f"  {_GREEN}{tid:2d}{_RESET}  {t['desc']}")
        print(f"      {_DIM}{t['prompt'][:80]}...{_RESET}\n")
    
    print(f"{_YELLOW}💡 PIPELINE GỢI Ý:{_RESET}")
    print(f"   Quick:     {_GREEN}son(df, [1,2,3]){_RESET}")
    print(f"   Basic:     {_GREEN}son(df, [1,2,3,4,5,6]){_RESET}")
    print(f"   Advanced:  {_GREEN}son(df, [1,2,3,4,5,7,8,6,10]){_RESET}")
    print(f"   ML Ready:  {_GREEN}son(df, [1,2,3,4,5,7,8,6,9,10]){_RESET}")
    print()


def _print_hd():
    
    print(f"""
{_BOLD}{_BLUE}{'='*70}{_RESET}
{_BOLD}{_BLUE}  🧠 PANDATOOLS - SƠN AI DataFrame Cleaner{_RESET}
{_BOLD}{_BLUE}  Version: 2.0 | Author: Sơn Lê{_RESET}
{_BOLD}{_BLUE}{'='*70}{_RESET}

{_BOLD}{_YELLOW}📋 1. HÀM PHÂN TÍCH DỮ LIỆU:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.intoo(){_RESET}")
    print(f"      {_DIM}Phân tích toàn bộ DataFrame, hiển thị bảng màu + gợi ý code{_RESET}")
    print(f"      {_DIM}Options: as_json=True → trả về JSON, max_rows=10{_RESET}")
    
    print(f"\n  {_GREEN}df.clean.summary(){_RESET}")
    print(f"      {_DIM}Tóm tắt thống kê: min, max, mean, skewness, unique{_RESET}")
    print(f"      {_DIM}Options: as_json=True → trả về JSON{_RESET}")
    
    print(f"\n  {_GREEN}df.clean.info_memory(){_RESET}")
    print(f"      {_DIM}Xem memory usage chi tiết từng cột{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}🔧 2. HÀM LÀM SẠCH THỦ CÔNG (KHÔNG CẦN API):{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    manual_funcs = [
        ["{_GREEN}df.clean.drop_dupes(){_RESET}", 
         "Xóa hàng trùng lặp",
         "subset=['col1','col2'], keep='first'/'last'/False, inplace=True"],
        
        ["{_GREEN}df.clean.strip_strings(){_RESET}", 
         "Strip khoảng trắng thừa string columns",
         "lowercase=True, inplace=True"],
        
        ["{_GREEN}df.clean.fix_dtypes(){_RESET}", 
         "Tự động sửa kiểu dữ liệu",
         "inplace=True"],
        
        ["{_GREEN}df.clean.fill_missing(){_RESET}", 
         "Điền giá trị thiếu thông minh",
         "numeric_strategy='median'/'mean', cat_fill='Unknown', datetime_strategy='ffill'/'bfill', add_indicator=True, inplace=True"],
        
        ["{_GREEN}df.clean.remove_uninformative(){_RESET}", 
         "Xóa cột >70% missing hoặc constant",
         "missing_threshold=0.7, inplace=True"],
        
        ["{_GREEN}df.clean.normalize_text(){_RESET}", 
         "Chuẩn hóa text Unicode NFC (tiếng Việt)",
         "columns=['col1','col2'], inplace=True"],
        
        ["{_GREEN}df.clean.clip_outliers(){_RESET}", 
         "Xử lý outliers bằng IQR",
         "strategy='iqr', multiplier=1.5, inplace=True"],
        
        ["{_GREEN}df.clean.normalize_column_names(){_RESET}", 
         "Chuẩn hóa tên cột (lowercase, space→_)",
         "inplace=True"],
        
        ["{_GREEN}df.clean.optimize_memory(){_RESET}", 
         "Tối ưu memory usage (downcast int/float)",
         "inplace=True"],
        
        ["{_GREEN}df.clean.auto(){_RESET}", 
         "Tự động làm sạch (gộp task 1-6)",
         "inplace=True"],
    ]
    
    for func, desc, params in manual_funcs:
        print(f"  {func}")
        print(f"      {_DIM}{desc}{_RESET}")
        print(f"      {_DIM}Params: {params}{_RESET}\n")
    
    print(f"""
{_BOLD}{_YELLOW}🤖 3. HÀM SƠN AI (CẦN API):{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.ai_son(model, key, provider){_RESET}")
    print(f"      {_DIM}Khởi tạo SƠN AI với model + API key (PHẢI GỌI TRƯỚC){_RESET}")
    print(f"      {_DIM}Params: model='gpt-4o', key='sk-...' hoặc key=1 (lấy từ env){_RESET}")
    
    print(f"\n  {_GREEN}df.clean.son([task_ids]){_RESET}")
    print(f"      {_DIM}Chạy SƠN AI pipeline với các task đã chọn{_RESET}")
    print(f"      {_DIM}VD: df.clean.son([1,2,4,7]) → chạy task 1,2,4,7{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}🐘 4. BIG DATA MODE:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.bigdata(start=True){_RESET}")
    print(f"      {_DIM}Bật chế độ xử lý dữ liệu lớn (tự động chunking, sampling){_RESET}")
    print(f"      {_DIM}Params: chunk_size=100000, sample_size=10000{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}📊 5. BIỂU ĐỒ:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.bd('col'){_RESET}            {_DIM}Tự động chọn biểu đồ phù hợp{_RESET}")
    print(f"  {_GREEN}df.clean.bd('x', 'y'){_RESET}         {_DIM}Tự detect loại biểu đồ 2 cột{_RESET}")
    print(f"  {_GREEN}df.clean.bd_bar(x, y){_RESET}         {_DIM}Bar chart{_RESET}")
    print(f"  {_GREEN}df.clean.bd_line(x, y){_RESET}        {_DIM}Line chart{_RESET}")
    print(f"  {_GREEN}df.clean.bd_pie(col){_RESET}          {_DIM}Pie chart{_RESET}")
    print(f"  {_GREEN}df.clean.bd_scatter(x, y){_RESET}     {_DIM}Scatter plot{_RESET}")
    print(f"  {_GREEN}df.clean.bd_hist(col){_RESET}         {_DIM}Histogram{_RESET}")
    print(f"  {_GREEN}df.clean.bd_box(x, y){_RESET}         {_DIM}Boxplot{_RESET}")
    print(f"  {_GREEN}df.clean.bd_heatmap(){_RESET}          {_DIM}Correlation heatmap{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}💾 6. LƯU FILE:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.csv('file.csv'){_RESET}        {_DIM}Lưu CSV (utf-8-sig){_RESET}")
    print(f"  {_GREEN}df.clean.excel('file.xlsx'){_RESET}     {_DIM}Lưu Excel{_RESET}")
    print(f"  {_GREEN}df.clean.to_parquet('file.parquet'){_RESET} {_DIM}Lưu Parquet (nhanh hơn CSV 10x){_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}📚 7. HÀM TIỆN ÍCH KHÁC:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_GREEN}df.clean.hd(){_RESET}              {_DIM}Hướng dẫn này{_RESET}")
    print(f"  {_GREEN}df.clean.list(){_RESET}            {_DIM}Xem danh sách SƠN AI task{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}🔑 8. KHỞI TẠO SƠN AI:{_RESET}
{_DIM}{'─'*70}{_RESET}

  {_CYAN}Cách 1: Key trực tiếp{_RESET}
  >>> df.clean.ai_son(model="gpt-4o", key="sk-abc123...")

  {_CYAN}Cách 2: Key từ env (vị trí 1, 2, 3){_RESET}
  >>> df.clean.ai_son(model="gpt-4o", key=1)

  {_CYAN}Cách 3: Key từ env mặc định{_RESET}
  >>> df.clean.ai_son(model="gpt-4o")

  {_CYAN}Cách 4: Dùng Gemini{_RESET}
  >>> df.clean.ai_son(model="gemini-pro", provider="gemini", key="...")

  {_DIM}Env key: OPENAI_API_KEY (1), SON_AI_KEY (2), AI_API_KEY (3){_RESET}

{_BOLD}{_YELLOW}📌 9. DANH SÁCH TASK:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    print(f"  {_BOLD}{'ID':<5} {'Icon':<5} {'Tên':<30} {'Chức năng'}{_RESET}")
    print(f"  {_DIM}{'─'*70}{_RESET}")
    
    for tid in sorted(TASKS):
        t = TASKS[tid]
        icon = {
            1: "🗑️", 2: "📝", 3: "✂️", 4: "🚫", 5: "🔧",
            6: "💉", 7: "📊", 8: "🔤", 9: "📅", 10: "💾"
        }.get(tid, "🔹")
        
        print(f"  {_GREEN}{tid:<5}{_RESET} {icon:<5} {t['desc']:<30} {_DIM}{t['prompt'][:50]}...{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}🚀 10. PIPELINE GỢI Ý:{_RESET}
{_DIM}{'─'*70}{_RESET}
""")
    
    pipelines = [
        ["Quick", "[1, 2, 3]", "Xóa trùng + Chuẩn hóa cột + Strip text"],
        ["Basic", "[1, 2, 3, 4, 5, 6]", "Làm sạch cơ bản đầy đủ"],
        ["Advanced", "[1, 2, 3, 4, 5, 7, 8, 6, 10]", "Có xử lý outliers + text"],
        ["ML Ready", "[1, 2, 3, 4, 5, 7, 8, 6, 9, 10]", "Sẵn sàng cho Machine Learning"],
    ]
    
    print(f"  {_BOLD}{'Pipeline':<15} {'Tasks':<40} {'Mô tả'}{_RESET}")
    print(f"  {_DIM}{'─'*70}{_RESET}")
    for name, tasks, desc in pipelines:
        print(f"  {_CYAN}{name:<15}{_RESET} {_GREEN}{tasks:<40}{_RESET} {_DIM}{desc}{_RESET}")
    
    print(f"""
{_BOLD}{_YELLOW}💡 11. VÍ DỤ ĐẦY ĐỦ:{_RESET}
{_DIM}{'─'*70}{_RESET}

  {_DIM}# Import{_RESET}
  import pandas as pd
  import pandatools

  {_DIM}# Đọc data{_RESET}
  df = pd.read_csv('data.csv')

  {_DIM}# Xem hướng dẫn{_RESET}
  df.clean.hd()

  {_DIM}# Phân tích{_RESET}
  df.clean.intoo()

  {_DIM}# Auto clean{_RESET}
  df = df.clean.auto()

  {_DIM}# Big Data{_RESET}
  df.clean.bigdata(start=True)
  df = df.clean.auto()

  {_DIM}# SƠN AI{_RESET}
  df.clean.ai_son(model="gpt-4o", key=1)
  df = df.clean.son([1, 2, 4, 7])

  {_DIM}# Biểu đồ{_RESET}
  df.clean.bd('age')
  df.clean.bd_bar('category', 'price', save='chart.png')

  {_DIM}# Lưu{_RESET}
  df.clean.csv('clean.csv')
  df.clean.to_parquet('clean.parquet')

{_BOLD}{_YELLOW}🎯 12. TỰ ĐỊNH NGHĨA TASK:{_RESET}
{_DIM}{'─'*70}{_RESET}

  @son_ai(11, "Prompt AI", "Mô tả")
  def my_task(df):
      return df

  df.clean.son([1, 2, 11])

{_BOLD}{_BLUE}{'='*70}{_RESET}
{_BOLD}{_BLUE}  📚 Hết | df.clean.hd() → xem lại | df.clean.list() → xem task{_RESET}
{_BOLD}{_BLUE}{'='*70}{_RESET}
""")


@pd.api.extensions.register_dataframe_accessor("clean")
class DataFrameCleaner:
    def __init__(self, df: pd.DataFrame):
        self._obj = df

    def _update(self, new_df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
        if inplace:
            if (new_df.shape == self._obj.shape and 
                list(new_df.columns) == list(self._obj.columns)):
                self._obj.loc[:, :] = new_df.values
            else:
                warnings.warn(f"⚠️ Shape mismatch: {self._obj.shape} → {new_df.shape}")
        return new_df

    def intoo(self, as_json: bool = False, max_rows: Optional[int] = None, indent: int = 2):
        df = self._obj
        
        if BigDataConfig.enabled and len(df) > BigDataConfig.sample_size:
            print(f"{_YELLOW}⚡ Big Data: sample {BigDataConfig.sample_size:,} dòng{_RESET}")
            df = df.sample(n=BigDataConfig.sample_size, random_state=42)
        
        columns = list(df.columns)
        if max_rows:
            columns = columns[:max_rows]

        records = []
        for col_name in columns:
            col = df[col_name]
            n_total = len(col)
            n_notnull = int(col.notna().sum())
            n_missing = n_total - n_notnull
            pct_miss = n_missing / n_total * 100 if n_total > 0 else 0.0
            dtype_str = str(col.dtype)
            issues = _analyze(col)

            records.append({
                "column": col_name,
                "dtype": dtype_str,
                "non_null": n_notnull,
                "total": n_total,
                "missing_count": n_missing,
                "missing_pct": round(pct_miss, 2),
                "issues": issues,
                "suggestion": _flat_suggest(issues),
                "code": _flat_code(issues),
            })

        if as_json:
            payload = {
                "shape": {"rows": len(self._obj), "cols": len(self._obj.columns)},
                "memory_mb": round(self._obj.memory_usage(deep=True).sum() / 1024**2, 1),
                "big_data_mode": BigDataConfig.enabled,
                "columns": records,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=indent))
            return records

        mem = self._obj.memory_usage(deep=True).sum()
        mem_str = f"{mem/1024:.1f} KB" if mem < 1024**2 else f"{mem/1024**2:.2f} MB"

        print(f"\n{_BOLD}{_CYAN}{'─'*72}{_RESET}")
        print(f"{_BOLD}📋 DataFrame Info – pandatools{_RESET}")
        print(f"{_DIM}{len(self._obj)} hàng × {len(self._obj.columns)} cột  |  Bộ nhớ: {mem_str}{_RESET}")
        if BigDataConfig.enabled:
            print(f"{_DIM}⚡ Phân tích trên sample {BigDataConfig.sample_size:,} dòng{_RESET}")
        print(f"{_BOLD}{_CYAN}{'─'*72}{_RESET}")

        display_rows = []
        for r in records:
            display_rows.append({
                "Cột": r["column"],
                "Non-Null": f"{r['non_null']}/{r['total']}",
                "Thiếu (%)": f"{r['missing_pct']:.1f}%",
                "Dtype": r["dtype"],
                "Gợi ý": r["suggestion"],
                "Code gợi ý": r["code"] if r["code"] else "—"
            })

        result_df = pd.DataFrame(display_rows)
        
        col_keys = ["Cột", "Non-Null", "Thiếu (%)", "Dtype", "Gợi ý", "Code gợi ý"]
        widths = {k: max(len(k), result_df[k].str.len().max()) for k in col_keys}
        widths["Code gợi ý"] = min(widths["Code gợi ý"], 60)

        header = "  ".join(k.ljust(widths[k]) for k in col_keys)
        print(f"{_BOLD}{header}{_RESET}")
        print("─" * (sum(widths.values()) + 2*(len(col_keys)-1)))

        for _, row in result_df.iterrows():
            sug = row["Gợi ý"]
            code = row["Code gợi ý"][:widths["Code gợi ý"]]

            if "DROP" in sug or "trùng" in sug or ">50%" in sug:
                sug_c = f"{_RED}{sug}{_RESET}"
                code_c = f"{_RED}{code}{_RESET}"
            elif sug == "✅ OK":
                sug_c = f"{_GREEN}{sug}{_RESET}"
                code_c = f"{_DIM}—{_RESET}"
            else:
                sug_c = f"{_YELLOW}{sug}{_RESET}"
                code_c = f"{_CYAN}{code}{_RESET}"

            miss = row["Thiếu (%)"]
            miss_v = float(miss.replace("%", ""))
            miss_c = (
                f"{_RED}{miss}{_RESET}" if miss_v > 50 else
                f"{_YELLOW}{miss}{_RESET}" if miss_v > 0 else
                f"{_GREEN}{miss}{_RESET}"
            )

            print(
                f"{str(row['Cột']).ljust(widths['Cột'])}  "
                f"{str(row['Non-Null']).ljust(widths['Non-Null'])}  "
                f"{miss_c.ljust(widths['Thiếu (%)'] + 10)}  "
                f"{str(row['Dtype']).ljust(widths['Dtype'])}  "
                f"{sug_c}  "
                f"{code_c}"
            )

        print(f"{_BOLD}{_CYAN}{'─'*72}{_RESET}\n")
        return pd.DataFrame(records)

    def summary(self, as_json: bool = False):
        df = self._obj
        rows = []
        for col in df.columns:
            s = df[col]
            issues = _analyze(s)
            row = {
                "Cột": col,
                "Dtype": str(s.dtype),
                "Missing %": f"{s.isna().mean()*100:.1f}%",
                "Unique": s.nunique(),
                "Gợi ý": _flat_suggest(issues)
            }
            if pd.api.types.is_numeric_dtype(s.dtype):
                row["Min"] = s.min()
                row["Max"] = s.max()
                row["Mean"] = round(float(s.mean()), 2)
                skew = s.skew() if len(s.dropna()) > 2 else 0
                row["Skew"] = round(skew, 2)
            else:
                row["Min"] = row["Max"] = row["Mean"] = row["Skew"] = "—"
            rows.append(row)
        
        result = pd.DataFrame(rows).set_index("Cột")
        
        if as_json:
            res_dict = result.reset_index().to_dict(orient='records')
            print(json.dumps(res_dict, ensure_ascii=False, indent=2))
            return res_dict
        
        print(f"\n{_BOLD}📊 SUMMARY{_RESET}\n")
        print(result.to_string())
        print()
        return result

    def info_memory(self):
        df = self._obj
        total = df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"\n{_BOLD}💾 MEMORY USAGE{_RESET}")
        print(f"  Total: {total:.1f} MB | Rows: {len(df):,} | Columns: {len(df.columns)}")
        if BigDataConfig.enabled:
            print(f"  {_YELLOW}🐘 Big Data Mode: ON | Chunk: {BigDataConfig.chunk_size:,}{_RESET}")
        
        mem_per_col = df.memory_usage(deep=True) / 1024**2
        top5 = mem_per_col.sort_values(ascending=False).head(5)
        print(f"\n  {_BOLD}Top 5 nặng nhất:{_RESET}")
        for col, mem in top5.items():
            print(f"    {col}: {mem:.1f}MB ({mem/total*100:.1f}%)")
        print()
        return self._obj

    def drop_dupes(self, subset=None, keep: str = "first", inplace: bool = False):
        df = self._obj.copy()
        n_before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"✅ drop_dupes: Xóa {_RED}{n_dropped}{_RESET} hàng ({n_before} → {len(df)})")
        return self._update(df, inplace)

    def strip_strings(self, lowercase: bool = False, inplace: bool = False):
        df = self._obj.copy()
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().replace(['', 'nan', 'None'], None)
            if lowercase:
                df[col] = df[col].str.lower()
        if len(str_cols):
            print(f"✅ strip_strings: Đã strip {_GREEN}{len(str_cols)}{_RESET} cột")
        return self._update(df, inplace)

    def fix_dtypes(self, inplace: bool = False):
        df = self._obj.copy()
        report = []
        for col in df.select_dtypes(include=['object']).columns:
            old = str(df[col].dtype)
            try:
                n = pd.to_numeric(df[col], errors='coerce')
                if n.notna().mean() > 0.9:
                    df[col] = n
                    report.append(f"{col}: {old} → numeric")
                    continue
            except: pass
            try:
                d = pd.to_datetime(df[col], errors='coerce', format='mixed')
                if d.notna().mean() > 0.9:
                    df[col] = d
                    report.append(f"{col}: {old} → datetime")
                    continue
            except: pass
            if df[col].nunique() <= 20:
                df[col] = df[col].astype('category')
                report.append(f"{col}: {old} → category")
        
        for col in df.select_dtypes(include=['float']).columns:
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                df[col] = df[col].astype('Int64')
                report.append(f"{col}: float → Int64")
        
        for col in df.select_dtypes(include=['int']).columns:
            if set(df[col].dropna().unique()).issubset({0, 1}):
                df[col] = df[col].astype(bool)
                report.append(f"{col}: int → bool")
        
        if report:
            print(f"✅ fix_dtypes: Đã sửa {_GREEN}{len(report)}{_RESET} cột")
        return self._update(df, inplace)

    def fill_missing(self, 
                     numeric_strategy: str = "median",
                     cat_fill: str = "Unknown",
                     datetime_strategy: str = "ffill",
                     add_indicator: bool = True,
                     inplace: bool = False):
        df = self._obj.copy()
        report = []
        for col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss == 0:
                continue
            
            pct_miss = n_miss / len(df)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if add_indicator and pct_miss > 0.1:
                    df[f"{col}_is_missing"] = df[col].isna().astype(int)
                fill_val = df[col].median() if numeric_strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_val)
                report.append(f"{col}: {n_miss} NaN → {numeric_strategy}")
                
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(method=datetime_strategy)
                report.append(f"{col}: {n_miss} NaN → {datetime_strategy}")
                
            else:
                mode_val = df[col].mode()
                label = mode_val[0] if not mode_val.empty else cat_fill
                df[col] = df[col].fillna(label)
                report.append(f"{col}: {n_miss} NaN → '{label}'")
        
        if report:
            print(f"✅ fill_missing: Đã fill {_GREEN}{len(report)}{_RESET} cột")
        return self._update(df, inplace)

    def remove_uninformative(self, missing_threshold: float = 0.7, inplace: bool = False):
        df = self._obj.copy()
        to_drop = []
        for col in df.columns:
            if df[col].isna().mean() > missing_threshold or df[col].nunique() <= 1:
                to_drop.append(col)
        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"✅ remove_uninformative: Xóa {_RED}{len(to_drop)}{_RESET} cột: {to_drop}")
        return self._update(df, inplace)

    def normalize_text(self, columns: List[str] = None, inplace: bool = False):
        import unicodedata
        df = self._obj.copy()
        target = columns or df.select_dtypes(include=['object', 'string']).columns
        for col in target:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: unicodedata.normalize('NFC', x) if pd.notna(x) else x
                )
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        if len(target):
            print(f"✅ normalize_text: Đã chuẩn hóa {_GREEN}{len(target)}{_RESET} cột")
        return self._update(df, inplace)

    def clip_outliers(self, strategy: str = "iqr", multiplier: float = 1.5, inplace: bool = False):
        df = self._obj.copy()
        report = []
        for col in df.select_dtypes(include=[np.number]).columns:
            non_null = df[col].dropna()
            if len(non_null) < 10:
                continue
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            n_out = ((df[col] < q1 - multiplier*iqr) | (df[col] > q3 + multiplier*iqr)).sum()
            if n_out > 0:
                df[col] = df[col].clip(q1 - multiplier*iqr, q3 + multiplier*iqr)
                report.append(f"{col}: {int(n_out)} outliers → clip")
        if report:
            print(f"✅ clip_outliers: Đã xử lý {_GREEN}{len(report)}{_RESET} cột")
        return self._update(df, inplace)

    def normalize_column_names(self, inplace: bool = False):
        df = self._obj.copy()
        old_cols = df.columns.tolist()
        new_cols = [str(c).strip().lower().replace(' ', '_').replace('-', '_')[:50] for c in old_cols]
        
        seen = {}
        final_cols = []
        for c in new_cols:
            if c in seen:
                seen[c] += 1
                final_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                final_cols.append(c)
        
        df.columns = final_cols
        changed = sum(1 for o, n in zip(old_cols, final_cols) if o != n)
        if changed:
            print(f"✅ normalize_column_names: Đổi {_GREEN}{changed}{_RESET} tên cột")
        return self._update(df, inplace)

    def optimize_memory(self, inplace: bool = False):
        df = self._obj.copy()
        initial = df.memory_usage(deep=True).sum() / 1024**2
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        final = df.memory_usage(deep=True).sum() / 1024**2
        print(f"✅ optimize_memory: {initial:.1f}MB → {_GREEN}{final:.1f}MB{_RESET} (giảm {(1-final/initial)*100:.1f}%)")
        return self._update(df, inplace)

    def auto(self, inplace: bool = False):
        mode = f"{_YELLOW}[BIG DATA]{_RESET} " if BigDataConfig.enabled else ""
        print(f"\n{_BOLD}{_MAGENTA}⚡ AUTO CLEAN {mode}(No API){_RESET}")
        
        df = self._obj.copy()
        for tid in [1, 2, 3, 4, 5, 6]:
            if tid in TASKS:
                try:
                    df = TASKS[tid]['fn'](df)
                    print(f"  {_GREEN}✅{_RESET} {TASKS[tid]['desc']}")
                except Exception as e:
                    print(f"  {_RED}❌{_RESET} {e}")
        
        if BigDataConfig.enabled:
            df = self.optimize_memory()
        
        print(f"{_GREEN}✅ Hoàn tất: {self._obj.shape} → {df.shape}{_RESET}\n")
        return self._update(df, inplace)

    def ai_son(self, model: str = "gpt-4o", key: Union[str, int] = None, provider: str = "openai"):
        api_key = _get_api_key(key)
        
        if not api_key:
            print(f"{_RED}❌ KHÔNG TÌM THẤY API KEY{_RESET}")
            print(f"{_YELLOW}💡 Cách khắc phục:{_RESET}")
            print(f"   1. Truyền key trực tiếp: df.clean.ai_son(key='sk-...')")
            print(f"   2. Set env: export OPENAI_API_KEY='sk-...'")
            print(f"   3. Dùng key=1, key=2, key=3 để lấy từ env")
            return self._obj
        
        SonConfig.model = model
        SonConfig.api_key = api_key
        SonConfig.provider = provider
        SonConfig.is_ready = True
        
        print(f"\n{_BOLD}{_GREEN}✅ SƠN AI ĐÃ SẴN SÀNG{_RESET}")
        print(f"   {_CYAN}Model:{_RESET}    {model}")
        print(f"   {_CYAN}Provider:{_RESET}  {provider}")
        print(f"   {_CYAN}API Key:{_RESET}   {api_key[:10]}...{api_key[-4:]}")
        print(f"\n{_DIM}   Dùng df.clean.son([...]) để chạy pipeline{_RESET}")
        print(f"{_DIM}   Dùng df.clean.list() để xem danh sách task{_RESET}\n")
        
        return self._obj

    def son(self, tasks: List[int] = None):
        if not SonConfig.is_ready:
            print(f"{_RED}❌ SƠN AI chưa khởi tạo!{_RESET}")
            print(f"{_YELLOW}👉 Dùng: df.clean.ai_son(model='gpt-4o', key='...'){_RESET}")
            return self._obj
        
        tasks = tasks or [1, 2, 3, 4, 5, 6]
        return son(self._obj, tasks)

    @staticmethod
    def list():
        list_tasks()

    def bigdata(self, start: bool = True, chunk_size: int = 100_000, sample_size: int = 10_000):
        if start:
            BigDataConfig.enabled = True
            BigDataConfig.chunk_size = chunk_size
            BigDataConfig.sample_size = sample_size
            BigDataConfig.low_memory = True
            
            print(f"\n{_BOLD}{_BLUE}{'='*60}{_RESET}")
            print(f"{_BOLD}{_BLUE}🐘 BIG DATA MODE: ON{_RESET}")
            print(f"{_BOLD}{_BLUE}{'='*60}{_RESET}")
            print(f"  Rows: {len(self._obj):,} | Memory: {self._obj.memory_usage(deep=True).sum()/1024**2:.1f}MB")
            print(f"  Chunk: {chunk_size:,} | Sample: {sample_size:,}")
            print(f"{_BOLD}{_BLUE}{'='*60}{_RESET}\n")
        else:
            BigDataConfig.enabled = False
            BigDataConfig.low_memory = False
            print(f"{_YELLOW}🐘 BIG DATA MODE: OFF{_RESET}")
        return self._obj

    def bd(self, x=None, y=None, z=None, kind=None, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib seaborn{_RESET}")
            return
        
        df = self._obj
        
        if x is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            cat_cols = df.select_dtypes(include=['object', 'category']).columns[:3]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[0, i])
                axes[0, i].set_title(f'{col}')
            for i, col in enumerate(cat_cols):
                df[col].value_counts().head(10).plot.pie(ax=axes[1, i], autopct='%1.1f%%')
                axes[1, i].set_title(f'{col}')
            plt.tight_layout()
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            if title: plt.suptitle(title, fontsize=16)
            plt.show()
            return
        
        col_x = x if isinstance(x, str) else df.columns[x]
        
        if y is None:
            if pd.api.types.is_numeric_dtype(df[col_x]):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.histplot(df[col_x].dropna(), kde=True, ax=axes[0])
                axes[0].set_title(f'Phân bố {col_x}')
                sns.boxplot(y=df[col_x].dropna(), ax=axes[1])
                axes[1].set_title(f'Boxplot {col_x}')
            elif pd.api.types.is_datetime64_any_dtype(df[col_x]):
                df[col_x].value_counts().sort_index().plot(figsize=(12, 5))
                plt.title(f'Tần suất {col_x}')
            else:
                df[col_x].value_counts().head(15).plot.bar(figsize=(12, 5))
                plt.title(f'Top 15 {col_x}')
        elif z is None:
            col_y = y if isinstance(y, str) else df.columns[y]
            if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.5)
                plt.title(f'{col_x} vs {col_y}')
            elif pd.api.types.is_datetime64_any_dtype(df[col_x]):
                plt.figure(figsize=(12, 5))
                plt.plot(df[col_x], df[col_y])
                plt.title(f'{col_y} theo {col_x}')
            else:
                plt.figure(figsize=(12, 6))
                df.groupby(col_x)[col_y].mean().sort_values().plot.bar()
                plt.title(f'{col_y} theo {col_x}')
        else:
            col_y = y if isinstance(y, str) else df.columns[y]
            col_z = z if isinstance(z, str) else df.columns[z]
            plt.figure(figsize=(12, 6))
            for name, group in df.groupby(col_z):
                plt.plot(group[col_x], group[col_y], label=name, alpha=0.7)
            plt.legend()
            plt.title(f'{col_y} theo {col_x}, nhóm {col_z}')
        
        if title: plt.title(title)
        if save: plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    def bd_bar(self, x, y=None, z=None, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            df = self._obj
            col_x = x if isinstance(x, str) else df.columns[x]
            col_y = y if isinstance(y, str) else df.columns[y] if y else None
            
            plt.figure(figsize=kwargs.get('figsize', (12, 6)))
            if col_y:
                if z:
                    col_z = z if isinstance(z, str) else df.columns[z]
                    df.pivot_table(values=col_y, index=col_x, columns=col_z, aggfunc='mean').plot.bar()
                else:
                    df.groupby(col_x)[col_y].mean().sort_values().plot.bar()
            else:
                df[col_x].value_counts().head(20).plot.bar()
            
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib{_RESET}")

    def bd_line(self, x, y=None, z=None, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            df = self._obj
            col_x = x if isinstance(x, str) else df.columns[x]
            col_y = y if isinstance(y, str) else df.columns[y] if y else None
            
            plt.figure(figsize=kwargs.get('figsize', (12, 5)))
            if col_y:
                if z:
                    col_z = z if isinstance(z, str) else df.columns[z]
                    for name, group in df.groupby(col_z):
                        plt.plot(group[col_x], group[col_y], label=name, alpha=0.7)
                    plt.legend()
                else:
                    plt.plot(df[col_x], df[col_y])
            else:
                df[col_x].value_counts().sort_index().plot()
            
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib{_RESET}")

    def bd_pie(self, col, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            df = self._obj
            col_name = col if isinstance(col, str) else df.columns[col]
            
            plt.figure(figsize=kwargs.get('figsize', (8, 8)))
            df[col_name].value_counts().head(10).plot.pie(autopct='%1.1f%%')
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib{_RESET}")

    def bd_scatter(self, x, y, z=None, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            df = self._obj
            col_x = x if isinstance(x, str) else df.columns[x]
            col_y = y if isinstance(y, str) else df.columns[y]
            
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            if z:
                col_z = z if isinstance(z, str) else df.columns[z]
                sns.scatterplot(data=df, x=col_x, y=col_y, hue=col_z, alpha=0.6)
            else:
                sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.5)
            
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib seaborn{_RESET}")

    def bd_hist(self, col, bins=30, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            df = self._obj
            col_name = col if isinstance(col, str) else df.columns[col]
            
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            sns.histplot(df[col_name].dropna(), kde=True, bins=bins)
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib seaborn{_RESET}")

    def bd_box(self, x, y=None, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            df = self._obj
            col_x = x if isinstance(x, str) else df.columns[x]
            
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            if y:
                col_y = y if isinstance(y, str) else df.columns[y]
                sns.boxplot(data=df, x=col_x, y=col_y)
            else:
                sns.boxplot(data=df[col_x].dropna())
            
            if title: plt.title(title)
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib seaborn{_RESET}")

    def bd_heatmap(self, save=None, title=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            df = self._obj
            
            plt.figure(figsize=kwargs.get('figsize', (12, 10)))
            sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
                       annot=True, fmt='.2f', cmap='coolwarm', center=0)
            if title: plt.title(title or 'Correlation Heatmap')
            if save: plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"{_RED}❌ Cần cài: pip install matplotlib seaborn{_RESET}")

    @staticmethod
    def hd():
        _print_hd()

    def csv(self, filename: str):
        if not filename.endswith('.csv'):
            filename += '.csv'
        try:
            self._obj.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ Đã lưu CSV: {_CYAN}{filename}{_RESET}")
        except Exception as e:
            print(f"{_RED}❌ Lỗi: {e}{_RESET}")
        return self._obj    def excel(self, filename: str):
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        try:
            self._obj.to_excel(filename, index=False)
            print(f"✅ Đã lưu Excel: {_CYAN}{filename}{_RESET}")
        except Exception as e:
            print(f"{_RED}❌ Lỗi: {e}{_RESET}")
        return self._obj

    def to_parquet(self, filename: str):
        if not filename.endswith('.parquet'):
            filename += '.parquet'
        try:
            self._obj.to_parquet(filename, index=False)
            size_mb = os.path.getsize(filename) / 1024**2
            print(f"✅ Đã lưu Parquet: {_CYAN}{filename}{_RESET} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"{_RED}❌ Lỗi: {e}{_RESET}")
        return self._obj


__all__ = [
    'son',
    'son_ai',
    'list_tasks',
    'SonConfig',
    'BigDataConfig',
    'DataFrameCleaner',
]
