import io, os, pickle, warnings
import pdfplumber
from docx import Document
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.generativeai as genai
import numpy as np

warnings.filterwarnings('ignore')
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# -------------------- المصادقة المحسنة --------------------
def authenticate_gdrive(use_console=False):
    creds = None
    token_file = "token.pickle"
    try:
        # احذف التوكن فقط إذا كان فاسداً، ليس دائماً
    if os.path.exists(token_file):
       with open(token_file, "rb") as token:
          creds = pickle.load(token)
    
      if creds and creds.valid:
          return build("drive", "v3", credentials=creds)  # استخدم التوكن الموجود

        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret_2_368639615599-s553j8nei3iolbq4as35abevl4ba6m61.apps.googleusercontent.com.json",
            SCOPES
        )

        if not use_console:   # لو شغال محلي
            creds = flow.run_local_server(
                port=0, prompt='consent', access_type='offline'
            )
        else:  # لو سيرفر (Streamlit Cloud)
            creds = flow.run_console(
                prompt='consent', access_type='offline'
            )

        with open(token_file, "wb") as token:
            pickle.dump(creds, token)

        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ خطأ في المصادقة: {e}")   # هنا
        if os.path.exists(token_file):
            os.remove(token_file)
        return None


# -------------------- الحصول على معلومات الحساب --------------------
def get_account_info(service):
    try:
        about = service.about().get(fields="user").execute()
        user_info = about.get('user', {})
        return {
            'email': user_info.get('emailAddress', 'غير معروف'),
            'name': user_info.get('displayName', 'غير معروف')
        }
    except Exception as e:
        print(f"❌ خطأ في الحصول على معلومات الحساب: {e}")
        return {'email': 'غير معروف', 'name': 'غير معروف'}

# -------------------- قراءة الملفات --------------------
def read_file(file_id, mime_type, service):
    text = ""
    try:
        if mime_type.startswith("application/vnd.google-apps"):
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        else:
            request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        if mime_type == "application/pdf":
            with pdfplumber.open(fh) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            temp_path = f"temp_{file_id}.docx"
            with open(temp_path, "wb") as f:
                f.write(fh.getvalue())
            doc = Document(temp_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            os.remove(temp_path)
        else:
            text = fh.read().decode("utf-8", errors='ignore')
    except:
        text = ""
    return text.strip()

# -------------------- فهرسة الملفات --------------------
def index_drive_files(service):
    results = service.files().list(
        pageSize=100, fields="files(id, name, mimeType)", q="trashed=false"
    ).execute()
    items = results.get("files", [])
    documents, file_names, file_ids = [], [], []
    allowed_types = [
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.spreadsheet',
        'application/vnd.google-apps.presentation',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    ]
    for item in items:
        if item["mimeType"] in allowed_types:
            text = read_file(item["id"], item["mimeType"], service)
            if len(text) > 20:
                documents.append(text)
                file_names.append(item["name"])
                file_ids.append(item["id"])
    return documents, file_names, file_ids

# -------------------- إنشاء Embeddings --------------------
def embed_texts(texts):
    genai.configure(api_key="AIzaSyAqH8V7huw-3R8CF7beqiwjUYk6PpNUc3E")
    embeddings = []
    for t in texts:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=t,
                task_type="retrieval_document"
            )
            embeddings.append(np.array(result['embedding']))
        except:
            embeddings.append(np.zeros(768))
    return embeddings

# -------------------- البحث المحسن --------------------
def search(query, documents, file_names, file_ids, doc_embeddings, top_k=3):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_emb = np.array(result['embedding'])
    except:
        return "", [], [], []
    
    sims = []
    for doc_emb in doc_embeddings:
        if np.linalg.norm(query_emb) > 0 and np.linalg.norm(doc_emb) > 0:
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            sims.append(sim)
        else:
            sims.append(0.0)
    
    min_similarity_threshold = 0.3
    filtered_indices = [i for i, score in enumerate(sims) if score >= min_similarity_threshold]
    if not filtered_indices:
        return "", [], [], []
    
    sorted_filtered = sorted(filtered_indices, key=lambda i: sims[i], reverse=True)
    top_idx = sorted_filtered[:top_k]
    
    context = "\n\n".join([documents[i] for i in top_idx])
    best_files = [file_names[i] for i in top_idx]
    best_file_ids = [file_ids[i] for i in top_idx]
    best_scores = [sims[i] for i in top_idx]
    
    return context, best_files, best_file_ids, best_scores

# -------------------- توليد الإجابة المحسنة --------------------
def answer_with_gemini(query, context, source_files):
    if not context.strip():
        return "❌ لا توجد ملفات تحتوي على إجابة لسؤالك في Drive."
    
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    prompt = f"""
بناءً على السياق التالي، أجب على السؤال بدقة ووضوح.
إذا لم تجد الإجابة في السياق، قل بصراحة أن المعلومات غير متوفرة.

السياق:
{context[:8000]}

السؤال: {query}

بعد الإجابة، أذكر المصادر التي استخدمتها من القائمة التالية:
{source_files}
"""
    try:
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") and resp.text else "⚠️ لم أستطع توليد إجابة"
    except Exception as e:
        return f"⚠️ خطأ في توليد الإجابة: {str(e)}"

# -------------------- الدالة الرئيسية --------------------
def main():
    print("🔐 جاري المصادقة مع Google Drive...")
    service = authenticate_gdrive(use_console=False)  # غيّريها True لو بتشغليه على سيرفر
    
    if not service:
        print("❌ فشلت المصادقة. تأكد من ملف credentials.")
        return
    
    account_info = get_account_info(service)
    print(f"✅ تمت المصادقة بنجاح! الحساب: {account_info['name']} ({account_info['email']})")
    
    print("📁 جاري فهرسة الملفات في Drive...")
    documents, file_names, file_ids = index_drive_files(service)
    if not documents:
        print("❌ لم يتم العثور على ملفات نصية قابلة للقراءة.")
        return
    
    print(f"✅ تم فهرسة {len(documents)} ملف")
    print("🔍 جاري إنشاء embeddings...")
    doc_embeddings = embed_texts(documents)
    print("✅ تم إنشاء embeddings بنجاح")
    
    while True:
        print("\n" + "="*50)
        query = input("💬 أدخل سؤالك (أو اكتب 'خروج' للإنهاء): ").strip()
        
        if query.lower() in ['خروج', 'exit', 'quit']:
            print("👋 مع السلامة!")
            break
        if not query:
            continue
            
        print("🔍 جاري البحث...")
        context, best_files, best_file_ids, best_scores = search(query, documents, file_names, file_ids, doc_embeddings)
        
        if not context.strip():
            print("❌ لا توجد ملفات تحتوي على إجابة لسؤالك في Drive.")
            continue
        
        print(f"📄 أفضل الملفات المطابقة ({len(best_files)}):")
        for i, (file, score) in enumerate(zip(best_files, best_scores)):
            print(f"  {i+1}. {file} (تشابه: {score:.2f})")
        
        print("🤖 جاري توليد الإجابة...")
        answer = answer_with_gemini(query, context, best_files)
        print(f"\n💡 الإجابة:\n{answer}")




