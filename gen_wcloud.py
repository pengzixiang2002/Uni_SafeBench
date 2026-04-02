import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud, STOPWORDS
import matplotlib.font_manager as fm  # <--- [新增] 引入字体管理模块

# ==========================================
# 【配置】字体路径与 PDF 格式
# ==========================================
# 指定字体文件的绝对路径
FONT_PATH = "./Uni-SafeBench/times.ttf"

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

# 【注意】移除了 plt.rcParams['font.serif'] = ['Times New Roman']
# 因为服务器系统库里没有这个字体会导致报错。我们将使用下面的 fontproperties 手动加载。

def generate_filtered_wordclouds():
    # ================= 配置路径 =================
    file_paths = [
        # VQA Tasks
        "./Uni-SafeBench/data/VQA/unsafety_I_unsafety_T/input.jsonl",
        "./Uni-SafeBench/data/VQA/safety_I_unsafety_T/input.jsonl",
        "./Uni-SafeBench/data/VQA/unsafety_I_safety_T/input.jsonl",
        "./Uni-SafeBench/data/VQA/safety_I_safety_T/input.jsonl",
        
        # Generation Tasks
        "./Uni-SafeBench/data/Text-Guided Image Editing/input.jsonl",
        "./Uni-SafeBench/data/Text Generation/inputs.jsonl",
        "./Uni-SafeBench/data/T2I/input.jsonl"
    ]

    output_dir = "wordcloud_results_icml"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ================= 名称映射 =================
    NAME_MAPPING = {
        "unsafety_I_unsafety_T": "Dual-Unsafe VQA",
        "safety_I_unsafety_T":   "Text-Unsafe VQA",
        "unsafety_I_safety_T":   "Image-Unsafe VQA",
        "safety_I_safety_T":     "Composite-Unsafe VQA",
        "T2I":                       "Unsafe T2I Generation",
        "Text-Guided Image Editing": "Malicious Image Editing",
        "Text Generation":           "Malicious Text Generation"
    }

    # ================= 排序逻辑 =================
    TARGET_ORDER = [
        "Dual-Unsafe VQA",         # (a)
        "Text-Unsafe VQA",         # (b)
        "Image-Unsafe VQA",        # (c)
        "Composite-Unsafe VQA",    # (d)
        "Malicious Image Editing",   # (e)
        "Malicious Text Generation", # (f)
        "Unsafe T2I Generation"      # (g)
    ]

    # ================= 停用词设置 =================
    custom_ignore_words = {
        "please", "pls", "kindly", "ask", "asked", "asking", "request", "help", "go","going","remove","change","take","seem","someone","caught","way","ways","others","train","dark","hand","personal","detailed","develop",
        "thank", "thanks", "wanted", "want","let","add","without","think","need","cause","getting","man","people","person","woman","really","next","might","now","look","step","recently","along","wants","guide","instruction","instructions","user","spread",
        "feel","everyone","said","high","card","number","modify","replace","friend","seems","demonstrate","illustrate","steps","method","animal","instruct","cat","bird","process","dog",
        "generate", "generated", "generating", "know","best",
        "write", "writing", "wrote","share","post","posts","looks",
        "create", "created", "creating",
        "make", "making", "made",
        "provide", "providing", "show", "showing", "give", "giving",
        "tell", "telling", "say", "saying", 
        "describe", "describing", "explain", "explaining",
        "list", "listed", "use", "using", "used",
        "prompt", "prompts", "input", "output", "image", "images", "picture", "pictures",
        "text", "texts", "question", "questions", "answer", "answers",
        "following", "sentence", "sentences", "word", "words", 
        "paragraph", "example", "examples", "description",
        "one", "two", "three", "don", "didn", "isn", "aren", "will", "would", "can", "could"
    }
    final_stopwords = set(STOPWORDS).union(custom_ignore_words)

    wc_config = {
        'width': 1000,
        'height': 500,
        'background_color': 'white',
        'max_words': 150,
        'contour_width': 1,
        'contour_color': 'steelblue',
        'stopwords': final_stopwords, 
        'collocations': False
    }

    print("-" * 60)
    print(f"{'Processing Task':<40} | {'Status'}")
    print("-" * 60)

    generated_results = []

    # ================= 生成词云 =================
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        path_parts = file_path.split('/')
        if 'VQA' in file_path:
            key_name = path_parts[-2]
        else:
            key_name = path_parts[-2]
        
        pretty_name = NAME_MAPPING.get(key_name, key_name)
        
        full_text = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        if 'prompt' in data:
                            full_text.append(data['prompt'].lower())
                    except json.JSONDecodeError:
                        continue
            
            text_combined = " ".join(full_text)
            if not text_combined:
                continue

            wc = WordCloud(**wc_config).generate(text_combined)
            generated_results.append({'pretty_name': pretty_name, 'wc': wc})
            print(f"{pretty_name:<40} | Generated")

        except Exception as e:
            print(f"{key_name:<40} | Error: {str(e)}")

    print("-" * 60)

    # ================= 绘图逻辑 =================
    num_plots = len(generated_results)
    if num_plots > 0:
        print("Generating ICML compliant PDF using custom font...")
        
        # 1. 加载自定义字体
        if os.path.exists(FONT_PATH):
            # 创建字体属性，指定大小为 16
            title_font = fm.FontProperties(fname=FONT_PATH, size=16)
            print(f"✅ Loaded font from: {FONT_PATH}")
        else:
            print(f"❌ Warning: Font file not found at {FONT_PATH}. Using default font.")
            title_font = None # 回退到默认

        # 2. 排序
        order_dict = {name: i for i, name in enumerate(TARGET_ORDER)}
        ordered_results = sorted(
            generated_results, 
            key=lambda x: order_dict.get(x['pretty_name'], 999)
        )

        # 3. 设置画布
        fig = plt.figure(figsize=(24, 7)) 
        
        gs = gridspec.GridSpec(2, 8, figure=fig, 
                               wspace=0.08, hspace=0.0, 
                               top=0.98, bottom=0.05, left=0.01, right=0.99)

        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        
        for i, item in enumerate(ordered_results):
            ax = None
            if i < 4:
                col_start = i * 2
                col_end = col_start + 2
                ax = fig.add_subplot(gs[0, col_start:col_end])
            else:
                row_idx = i - 4
                col_start = 1 + (row_idx * 2)
                col_end = col_start + 2
                ax = fig.add_subplot(gs[1, col_start:col_end])

            ax.imshow(item['wc'], interpolation='bilinear')
            ax.axis('off')
            
            # ==========================================
            # 【关键】应用自定义字体
            # ==========================================
            if title_font:
                # 使用 fontproperties 参数
                ax.text(0.5, -0.03, f"{labels[i]} {item['pretty_name']}", 
                        transform=ax.transAxes, 
                        ha='center', va='top', 
                        fontproperties=title_font)
            else:
                # 如果没找到文件，回退到普通方式
                ax.text(0.5, -0.03, f"{labels[i]} {item['pretty_name']}", 
                        transform=ax.transAxes, 
                        ha='center', va='top', fontsize=16, weight='normal')

        # 保存 PDF
        output_filename = "combined_wordclouds_icml.pdf"
        combined_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(combined_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        print(f"Combined PDF saved to: {combined_path}")
    else:
        print("No wordclouds were generated.")

if __name__ == "__main__":
    generate_filtered_wordclouds()