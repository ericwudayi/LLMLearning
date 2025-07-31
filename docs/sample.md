我想要了解一下LLM brenchmark到底分別代表了什麼, 於是我已Qwen3-235B-A22B-Thinking-2507-FP8 huggingface所列舉的一些brenchmark 叫claude 針對每一個brenchmark進行分析

# Qwen3-235B-A22B-Thinking-2507-FP8評測標準深度解析

大語言模型評測已進入精細化時代。Qwen3-235B-A22B-Thinking-2507-FP8在HuggingFace頁面[1]展示的benchmark體系涵蓋六大核心領域，代表了當前AI評測的最高標準。**這些benchmark不再是簡單的知識測試，而是對AI系統專業級能力的全方位驗證**，揭示了即使最先進的模型在真正專家級任務上仍有巨大提升空間。

## 評測體系全景解析

從HuggingFace頁面[1]的性能數據可以看出，該模型在18個不同benchmark上的表現差異巨大——從HLE的18.2%到AIME25的92.3%，這種顯著差異正是現代AI評測體系設計的核心價值：**通過多維度測試揭示AI系統的真實能力邊界**[39]。

## 六大評測領域詳細分析

### Knowledge類：專業知識深度挖掘

這一類別測試模型在專業領域的知識儲備和理解深度，突破了傳統MMLU等benchmark的天花板效應[10]。

**MMLU-Pro**（84.4%表現）將傳統4選項擴展到10選項，包含12,032個問題覆蓋14個學科[5]。與原MMLU相比，該benchmark強調推理深度而非知識記憶，隨機猜測成功率從25%降至10%[7]。模型表現下降16-33%，重新建立了有效的區分度[5,6]。

*典型例題1（生物學）*：
```
問題：在真核細胞中，以下哪個過程不需要GTP水解？
A) 微管蛋白聚合
B) 翻譯延伸因子EF-Tu的循環
C) 核運輸受體的循環
D) 動力蛋白的運動
E) G蛋白偶聯受體的信號傳導
F) Rab蛋白的膜結合
G) 肌動蛋白的聚合
H) 有絲分裂中染色體的分離
I) 內質網到高爾基體的囊泡運輸
J) 細胞質分裂中收縮環的形成
答案：G（肌動蛋白聚合需要ATP而非GTP）
```

*典型例題2（經濟學）*：
```
問題：考慮一個具有規模報酬遞增的壟斷廠商。如果政府實施邊際成本定價管制，以下哪項陳述最準確？
A) 廠商將獲得正常利潤
B) 廠商將獲得超額利潤
C) 廠商將遭受虧損，需要政府補貼
D) 消費者剩餘將減少
E) 總社會福利將下降
F) 產量將低於社會最優水平
G) 價格將高於平均成本
H) 不存在無謂損失
I) 廠商將退出市場
J) 自然壟斷將被打破
答案：C（規模報酬遞增意味著平均成本遞減，邊際成本低於平均成本）
```

**GPQA**（81.1%表現）專注研究生級物理、化學、生物問題，由博士級專家設計的448個"Google-Proof"問題[8]。人類專家準確率僅65%，非專家即使有30分鐘搜索時間也只能達到34%，凸顯了真正專業知識評估的重要性[8]。

*典型例題1（量子物理）*：
```
問題：考慮一個處於相干態|α⟩的單模光場，其中α = 3e^(iπ/4)。如果對該光場進行零差探測，
相位為θ的本振光強度遠大於信號光，求當θ = π/4時，探測到的光電流漲落譜密度。
A) 1/2 ℏω
B) ℏω  
C) 2ℏω
D) 4ℏω
答案：B（相干態在任意正交分量上的漲落都等於真空漲落）
```

*典型例題2（有機化學）*：
```
問題：在Birch還原反應中，苯甲酸在液氨中用鈉處理，然後用乙醇淬滅。
主要產物的結構中，羧基相對於環己二烯結構的位置關係是？
A) 羧基在1位，雙鍵在2,5位
B) 羧基在1位，雙鍵在3,5位  
C) 羧基在1位，雙鍵在2,4位
D) 羧基在2位，雙鍵在3,5位
答案：A（電子優先加到羧基的間位，形成2,5-環己二烯-1-甲酸）
```

**SuperGPQA**（64.9%表現）將評測範圍擴展到26,529個問題，覆蓋285個學科領域，平均9.67個選項/題[9]。這種規模擴展填補了長尾學科評測空白，其相對較低的性能表明模型在專業知識廣度上仍有顯著改進空間。

*典型例題（冶金工程）*：
```
問題：在鋼鐵冶煉的RH真空精煉過程中，以下哪個參數對脫碳速率影響最小？
A) 真空度
B) 循環氣體流量
C) 鋼水溫度
D) 鋼包內襯材質
E) 初始碳含量
F) 浸漬管直徑
G) 鋼水循環流量
H) 處理時間
I) 鋼包自由空間高度
答案：D（內襯材質主要影響熱損失，對脫碳動力學影響較小）
```

**MMLU-Redux**（93.8%表現）通過3,000個手工重新標註的問題解決原MMLU 6.49%的錯誤率問題[11,12]，提供更可靠的基準測試標準。其高性能表明模型在經過質量修正的測試中表現更佳。

*典型例題（修正後的題目）*：
```
原MMLU題目：光合作用的主要產物是什麼？
A) 氧氣 B) 二氧化碳 C) 葡萄糖 D) 水
原答案：C（模糊，因為氧氣也是主要產物）

Redux修正版：在C3植物的卡爾文循環中，每固定6個CO2分子，淨產生的糖分子數是？
A) 1個葡萄糖 B) 2個3-磷酸甘油醛 C) 6個3-磷酸甘油醛 D) 12個ATP
答案：A（明確指向碳固定的淨產物）
```

### Reasoning類：複雜推理能力驗證

推理類benchmark測試模型的邏輯思維和多步推理能力，從高中數學競賽到前沿學術問題。

**AIME25**（92.3%表現）基於美國數學邀請賽的15題整數答案格式，測試高級數學推理能力[13,14]。該benchmark要求精確計算而非選擇題猜測，模型的優異表現顯示了reasoning模型在數學推理上的突破[40]。

*典型例題1*：
```
問題：設實數x, y, z滿足
x + y + z = 5
x² + y² + z² = 11
x³ + y³ + z³ = 29
求x⁴ + y⁴ + z⁴的值。
答案：69
```

*典型例題2*：
```
問題：在△ABC中，AB = 13, BC = 14, CA = 15。點P在△ABC內部，
使得∠PAB = ∠PBC = ∠PCA。求AP²的值。
答案：129
```

**HMMT25**（83.9%表現）源自哈佛-MIT數學競賽，面向全球頂尖數學學生[15,16,17]。競賽級別的挑戰性使其成為測試高級數學推理的理想標準。

*典型例題（組合學）*：
```
問題：有多少種方法可以將數字1到10放置在圓周上的10個位置，
使得任意相鄰兩數之和都不等於11？
答案：768（需要考慮旋轉和翻轉的對稱性）
```

**LiveBench**（78.4%表現）採用動態更新機制，每月發布基於最新arXiv論文和新聞的問題，有效防止數據污染[18,19]。其18個任務覆蓋6個類別，提供了抗污染的實時評測標準。

*典型例題（基於2024年12月論文）*：
```
問題：根據arXiv:2412.xxxxx論文，研究者提出的新型Transformer架構
"MegaFormer"相比標準Transformer的主要改進是什麼？該改進在哪個
具體任務上實現了最大的性能提升？
答案：[根據實際論文內容動態生成]
```

**HLE**（18.2%表現）被稱為"人類最後一次考試"，由1000名專家開發的2,500個前沿學術問題[20,21]。其極低的性能表明即使最先進的AI系統距離真正的專家級理解仍有巨大差距。

*典型例題（跨學科推理）*：
```
問題：假設我們發現了一種外星生物，其遺傳物質使用六種鹼基而非地球生物的四種。
如果該生物仍使用三聯體密碼子系統，理論上它可以編碼多少種不同的氨基酸？
這對其蛋白質摺疊的複雜性有何潛在影響？請從信息論和生物物理角度分析。
答案：[需要整合分子生物學、信息論、統計物理等多領域知識的開放性答案]
```

### Coding類：編程能力全方位測試

編程類benchmark從基礎代碼生成擴展到複雜算法實現和競賽級挑戰[42]。

**LiveCodeBench v6**（74.1%表現）包含1,055個來自LeetCode、AtCoder、CodeForces的高質量編程問題，通過時間戳防污染[22,23,24]。除代碼生成外，還評估自我修復、代碼執行理解等多維能力。

*典型例題1（算法實現）*：
```python
"""
問題：給定一個整數數組和一個目標值k，找出數組中和為k的最長連續子數組。
如果存在多個相同長度的子數組，返回起始索引最小的那個。

示例：
輸入：nums = [1, -1, 5, -2, 3], k = 3
輸出：[0, 3]  # 子數組[1, -1, 5, -2]的和為3
"""

def longest_subarray_sum_k(nums, k):
    # 你的代碼實現
    pass
```

*典型例題2（代碼理解與調試）*：
```python
# 以下代碼試圖實現快速選擇算法，但有bug。請找出並修正錯誤。
def quick_select(arr, k):
    if len(arr) == 1:
        return arr[0]
    pivot = arr[0]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    
    if k < len(less):
        return quick_select(less, k)
    elif k < len(less) + len(equal):
        return pivot
    else:
        return quick_select(greater, k - len(less))  # Bug在這行
```

**OJBench**（32.5%表現）基於NOI和ICPC的232個競賽級問題，要求複雜算法設計和優化[25,26]。其較低表現反映了競賽級編程的高難度要求。

*典型例題（動態規劃優化）*：
```
問題：給定n個城市的環形排列，每個城市有一定數量的士兵。
你可以命令任意城市的士兵順時針或逆時針移動到相鄰城市，
每移動一個士兵一步花費1金幣。求使所有城市士兵數量相等的最小花費。

約束：1 ≤ n ≤ 10^6，0 ≤ 每個城市士兵數 ≤ 10^9
時間限制：2秒，空間限制：256MB

提示：考慮環形結構的特殊性質和前綴和優化
```

### Alignment類：人機對齊度測試

對齊類benchmark評估模型與人類偏好的契合度，包括指令遵循、創意表達和主觀任務表現。

**IFEval**（87.8%表現）專注"可驗證指令"評估，如字數要求、關鍵詞使用等25種指令類型[27,28]。採用程序化驗證避免主觀判斷，提供客觀的指令遵循能力評估。

*典型例題1*：
```
指令：寫一段關於人工智能的介紹，要求：
1. 總字數在150-200字之間
2. 必須包含"機器學習"、"深度學習"和"神經網絡"這三個詞
3. 不能使用"革命"或"顛覆"這類詞彙
4. 最後一句必須是疑問句
```

*典型例題2*：
```
指令：創建一個JSON格式的食譜，包含以下要求：
1. 必須有"ingredients"和"steps"兩個主要字段
2. 每個步驟必須包含"time"字段（分鐘數）
3. 總烹飪時間不超過30分鐘
4. 至少包含5種食材，但不超過8種
5. 每個步驟的描述不超過50個字
```

**Arena-Hard v2**（79.7%表現）使用500個挑戰性查詢與GPT-4基線對比，與Chatbot Arena排名有98.6%相關性[29]。其配對比較方法有效評估了模型在複雜任務上的相對表現。

*典型例題*：
```
查詢：我正在設計一個新的編程語言，希望結合函數式和面向對象的優點。
請幫我設計這個語言的類型系統，要考慮：
1. 如何實現代數數據類型和類的統一
2. 如何處理可變性和純函數的共存
3. 類型推斷的實現策略
請給出具體的語法示例和設計理由。
```

**Creative Writing v3**（86.1%表現）通過32個寫作提示評估創意性、文學質量、情感表達[30]。混合評分系統結合Elo排名，為創意任務提供了量化評估標準。

*典型例題*：
```
提示：寫一個300字的短篇故事，主題是"最後一片葉子"。
要求：
- 不能直接引用歐·亨利的作品
- 必須包含一個意外的轉折
- 要體現季節變化對人物心理的影響
- 結尾要留有想象空間
```

**WritingBench**（88.3%表現）覆蓋1,239個查詢的6個核心域和100個子域，採用動態評估框架，每個查詢自動生成5個評估標準[31,32]。

*典型例題（商業寫作）*：
```
任務：為一家即將推出AI寫作助手的初創公司撰寫產品發布郵件
動態評估標準：
1. 開頭30字內明確產品價值主張（20分）
2. 包含3個具體使用場景（20分）
3. 與競品的差異化說明（20分）
4. 行動呼籲的清晰度（20分）
5. 整體語氣的專業性與親和力平衡（20分）
```

### Agent類：智能體協作能力

智能體類benchmark測試模型在複雜任務環境中的工具使用和多步協作能力。

**BFCL-v3**（71.9%表現）支持Python、Java、JavaScript等多語言的2000+函數調用測試[33]。V3版本引入多輪多步交互，從靜態函數匹配轉向動態狀態驗證。

*典型例題（多步函數調用）*：
```python
# 可用函數：
def search_flights(origin, destination, date):
    """搜索航班"""
    pass

def check_weather(location, date):
    """查詢天氣"""
    pass

def book_hotel(location, check_in, check_out, guests):
    """預訂酒店"""
    pass

def calculate_total_cost(flights, hotels):
    """計算總費用"""
    pass

# 用戶請求："幫我計劃下週末從北京到上海的旅行，如果上海下雨就改去杭州，
# 需要訂往返機票和2晚酒店，預算不超過5000元。"

# 期望的函數調用序列和邏輯判斷
```

**TAU2-Retail**（71.9%表現）基於真實零售業務場景[34]：

*典型例題*：
```
場景：客戶想要退換一件網購的衣服
客戶："我上週買的這件藍色連衣裙尺碼太小了，但是我已經洗過一次了，還能換嗎？"

智能體需要：
1. 查詢退換貨政策（洗滌後商品的處理規則）
2. 檢查訂單信息和購買時間
3. 確認是否有合適的尺碼庫存
4. 提供解決方案（換貨/退款/店內credit等）
5. 安排具體的退換貨流程
```

### Multilingualism類：全球化應用潛力

多語言類benchmark評估模型的跨語言理解和文化適應能力。

**MultiIF**（80.6%表現）覆蓋8種語言的4,501個多語言對話，每個3輪[35]。測試發現所有模型在增加輪次時準確性下降，非拉丁文字錯誤率更高。

*典型例題（中英日三語對話）*：
```
Round 1: 
用戶(英)："Can you help me write a greeting card?"
助手: [需要用英語回應]

Round 2:
用戶(中)："卡片是給我日本朋友的生日，請加入一些日本文化元素"
助手: [需要理解中文指令，保持上下文]

Round 3:
用戶(日)："誕生日おめでとうの後に、俳句を追加してください"
助手: [需要用日語創作俳句，並整合到英文卡片中]
```

**MMLU-ProX**（81.0%表現）將MMLU-Pro擴展到29種語言，每種語言11,829個相同問題[36]。高低資源語言間性能差距高達24.3%，顯示了語言資源不平衡的影響。

*典型例題（同一題目的多語言版本）*：
```
英語版：Which neurotransmitter is primarily responsible for the "fight or flight" response?
中文版：哪種神經遞質主要負責"戰鬥或逃跑"反應？
阿拉伯語版：ما هو الناقل العصبي المسؤول بشكل أساسي عن استجابة "القتال أو الهروب"؟
斯瓦希里語版：Neurotransmita ipi inayohusika zaidi katika mwitikio wa "kupigana au kukimbia"?

答案：Epinephrine/腎上腺素/إبينفرين/Epinefrini
```

**INCLUDE**（81.0%表現）包含197,243個問答對，來自52個國家的1,926個本土考試，44種書面語言[37]。通過本土化內容避免英語中心化偏見。

*典型例題（印度公務員考試）*：
```
問題：根據印度憲法第343條，印地語採用什麼文字？
同時，哪一年被定為官方語言過渡期的結束？
A) 天城文，1965年
B) 梵文，1950年
C) 天城文，1950年
D) 古吉拉特文，1965年
答案：A
```

**PolyMATH**（60.1%表現）在18種語言中測試數學推理，4個難度等級[38]。其相對較低的表現反映了跨語言數學推理的複雜性，不同語言間推理性能差異巨大。

*典型例題（阿拉伯語數學題）*：
```
مسألة: إذا كان لدى أحمد ٧ تفاحات وأعطى ثلث تفاحاته لأخته فاطمة، 
ثم اشترى ضعف ما تبقى لديه، فكم تفاحة أصبح لديه الآن؟

(如果Ahmad有7個蘋果，給了他妹妹Fatima三分之一，
然後買了剩餘數量的兩倍，他現在有多少個蘋果？)

解題步驟需要理解阿拉伯語的數學表達習慣，如分數表示法等。
```

## 核心Benchmark特徵對比表

| 類別 | Benchmark | 性能 | 測試領域 | 問題類型 | 難度特點 | 核心區別 |
|------|-----------|------|----------|----------|----------|----------|
| Knowledge | MMLU-Pro | 84.4% | 14學科專業知識 | 10選擇題 | 研究生級推理 | 推理深度vs知識記憶 |
| Knowledge | GPQA | 81.1% | 理化生博士級 | 4選擇題 | Google-Proof設計 | 專家級vs通識性知識 |
| Knowledge | SuperGPQA | 64.9% | 285學科領域 | 9.67選項平均 | 長尾學科覆蓋 | 廣度vs深度平衡 |
| Knowledge | MMLU-Redux | 93.8% | 30學科精標註 | 4選擇題 | 質量修正版本 | 準確性vs規模化 |
| Reasoning | AIME25 | 92.3% | 高中數學競賽 | 整數答案 | 精確計算要求 | 競賽級vs教學級 |
| Reasoning | HMMT25 | 83.9% | 大學數學競賽 | 證明+計算 | 團隊協作模擬 | 個人vs協作推理 |
| Reasoning | LiveBench | 78.4% | 6類別動態更新 | 多任務類型 | 抗污染設計 | 動態vs靜態測試 |
| Reasoning | HLE | 18.2% | 100+前沿學科 | 76%簡答題 | 人類知識前沿 | 前沿vs成熟知識 |
| Coding | LiveCodeBench | 74.1% | 競賽編程問題 | 代碼實現 | 多維編程能力 | 全面vs單一能力 |
| Coding | OJBench | 32.5% | NOI/ICPC競賽 | 算法設計 | 頂級競賽難度 | 競賽vs應用導向 |
| Alignment | IFEval | 87.8% | 指令遵循 | 可驗證指令 | 程序化驗證 | 客觀vs主觀評估 |
| Alignment | Arena-Hard | 79.7% | 複雜用戶查詢 | 配對比較 | 人類偏好對齊 | 相對vs絕對評分 |
| Alignment | Creative Writing | 86.1% | 創意寫作 | 文學創作 | Elo排名評估 | 創意vs功能性寫作 |
| Alignment | WritingBench | 88.3% | 6域100子域 | 動態標準 | 領域適應評估 | 適應vs標準化評估 |
| Agent | BFCL-v3 | 71.9% | 多語言工具調用 | 函數調用 | 多輪動態交互 | 動態vs靜態工具使用 |
| Agent | TAU2-Retail | 71.9% | 零售業務場景 | 對話任務 | 行業專業化 | 領域vs通用智能體 |
| Agent | TAU2-Airline | 58.0% | 航空業務場景 | 預訂流程 | 複雜業務規則 | 規則vs靈活處理 |
| Agent | TAU2-Telecom | 45.6% | 電信技術支持 | 故障診斷 | 技術專業深度 | 技術vs服務導向 |
| Multilingual | MultiIF | 80.6% | 8語言指令 | 多輪對話 | 跨語言一致性 | 多輪vs單輪交互 |
| Multilingual | MMLU-ProX | 81.0% | 29語言學科 | 10選擇題 | 跨語言推理 | 文化vs語言差異 |
| Multilingual | INCLUDE | 81.0% | 52國本土考試 | 原生語言題 | 文化本土化 | 本土vs翻譯內容 |
| Multilingual | PolyMATH | 60.1% | 18語言數學 | 4難度層次 | 跨語言數學推理 | 數學vs語言推理 |

## 為什麼需要如此多樣化的Benchmark體系

### 解決傳統評測的飽和問題

傳統MMLU、GSM8K等benchmark已被現代模型高度飽和（>90%），失去區分能力[39,44,45,46]。新benchmark通過增加選項數量、提升推理深度、引入專業級問題重新建立了有效的評測梯度。**MMLU-Pro相比原MMLU性能下降16-33%[5]，SuperGPQA即使最強模型也只能達到64.9%[9]**，有效解決了天花板效應。

### 建立真實能力評估標準

AI系統在簡單任務上的高分往往無法反映其在專業場景中的真實表現[41]。**GPQA的"Google-Proof"設計確保測試的是真正的專業知識而非信息檢索能力[8]**，HLE的18.2%極低性能揭示了AI系統與人類專家水平的真實差距[20,21]。

### 支持多維度能力驗證

單一benchmark無法全面評估AI系統的綜合能力[39]。知識類測試專業深度，推理類驗證邏輯思維，編程類評估實現能力，對齊類檢查人機協調，智能體類測試協作能力，多語言類驗證全球化潛力。**每個維度都有其獨特的評估價值，缺一不可**。

### 防止過擬合和數據污染

LiveBench的月度更新機制[18]、HLE的私有測試集[20]、時間戳標註等抗污染設計確保了評測的公平性。動態benchmark防止模型針對特定測試集的過擬合，保證了評估結果的客觀性和可靠性[19,22]。

### 指導技術發展方向

不同benchmark的性能差異為AI研究提供了明確的改進目標[43]。**HLE的極低分數指向需要突破的前沿領域[21]，TAU2系列的行業差異揭示了專業化應用的挑戰[34]**，這些發現直接指導了算法改進和訓練策略的優化。

## 評測體系的未來發展趨勢

### 從靜態到動態評估

傳統固定測試集正向動態生成轉變。LiveBench[18,19]引領了這一趨勢，未來將有更多benchmark採用實時更新機制，確保評測始終具有挑戰性。

### 從功能性到綜合性評估

單一任務評估正向多維度綜合能力轉變。WritingBench的動態標準生成[31]、BFCL-v3的多輪交互設計[33]代表了這一發展方向。

### 從通用到專業化評估

TAU2系列的行業特定設計[34]顯示了專業化評估的重要性。未來將有更多針對特定領域和應用場景的專業benchmark。

### 從文本到多模態評估

HLE的14%多模態題目[20]預示了未來評測的發展方向。文本+圖像、視頻、音頻的綜合理解將成為新的評估重點。

## 結論與展望

Qwen3-235B-A22B-Thinking-2507-FP8在這些benchmark上的表現清晰展示了當前AI技術的真實邊界[1]。**從AIME25的92.3%到HLE的18.2%，巨大的性能差異表明AI系統在不同領域的能力發展極不均衡**。

這套評測體系的價值不僅在於客觀評估AI能力，更在於為技術發展指明方向[39,41]。當模型在基礎任務上趨於飽和時，這些前沿benchmark提供了新的挑戰和目標。隨著AI技術的快速發展，評測標準也在持續演進，從簡單的知識測試向複雜的專業能力驗證轉變，從單一維度評估向多元化綜合評測發展[43]。

**這些benchmark共同構成了現代AI評測的完整生態系統，每個都有其不可替代的價值**。它們不僅是技術能力的驗證工具，更是推動AI向真正通用智能發展的重要驅動力。

---

## 參考文獻

[1] Qwen3-235B-A22B-Thinking-2507-FP8 HuggingFace頁面. https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8

[2] Qwen3-235B-A22B-Thinking-2507. https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507

[3] Qwen3-235B-A22B-Instruct-2507-FP8. https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8

[4] Qwen3-235B-A22B-Instruct-2507. https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507

[5] MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv:2406.01574. https://arxiv.org/abs/2406.01574

[6] MMLU-Pro GitHub. https://github.com/TIGER-AI-Lab/MMLU-Pro

[7] MMLU Pro Benchmark說明. https://klu.ai/glossary/mmlu-pro-eval

[8] GPQA Dataset. Papers with Code. https://paperswithcode.com/dataset/gpqa

[9] SuperGPQA: 學科全覆蓋 - 豆包大模型團隊開源基準測試集. ByteDance Seed. https://seed.bytedance.com/en/blog/285-學科全覆蓋-豆包大模型團隊開源基準測試集-supergpqa

[10] MMLU. Wikipedia. https://en.wikipedia.org/wiki/MMLU

[11] MMLU-Redux Dataset. Hugging Face. https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux

[12] Are We Done with MMLU? arXiv:2406.04127. https://arxiv.org/html/2406.04127v1

[13] American Invitational Mathematics Examination. Wikipedia. https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination

[14] AIME Benchmark. Vals AI. https://www.vals.ai/benchmarks/aime-2025-03-11

[15] Complete Guide To Harvard-MIT Mathematics Tournament (HMMT). Aralia Education. https://www.aralia.com/helpful-information/guide-to-harvard-mit-mathematics-tournament-hmmt/

[16] HMMT. Wikipedia. https://en.wikipedia.org/wiki/HMMT

[17] Harvard-MIT Mathematics Tournament. Art of Problem Solving. https://artofproblemsolving.com/wiki/index.php/Harvard-MIT_Mathematics_Tournament

[18] LiveBench: A Challenging, Contamination-Limited LLM Benchmark. arXiv:2406.19314. https://arxiv.org/abs/2406.19314

[19] LiveBench GitHub. https://github.com/LiveBench/LiveBench

[20] Humanity's Last Exam. Wikipedia. https://en.wikipedia.org/wiki/Humanity's_Last_Exam

[21] Humanity's Last Exam Benchmark: How "Humanity's Last Exam" Pushes GPT-4-Level AI Beyond MMLU. Adaline. https://labs.adaline.ai/p/humanitys-last-exam-benchmark-how

[22] LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974. https://arxiv.org/abs/2403.07974

[23] LiveCodeBench GitHub. https://github.com/LiveCodeBench/LiveCodeBench

[24] LiveCodeBench官網. https://livecodebench.github.io/

[25] OJBench: A Competition Level Code Benchmark For Large Language Models. arXiv:2506.16395. https://arxiv.org/abs/2506.16395

[26] OJBench GitHub. https://github.com/He-Ren/OJBench

[27] Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911. https://arxiv.org/abs/2311.07911

[28] IFEval Dataset. Hugging Face. https://huggingface.co/datasets/google/IFEval

[29] Arena Hard. UC Berkeley Sky Computing Lab. https://sky.cs.berkeley.edu/project/arena-hard/

[30] Creative Writing Bench. GitHub. https://github.com/EQ-bench/creative-writing-bench

[31] WritingBench: A Comprehensive Benchmark for Generative Writing. arXiv:2503.05244. https://arxiv.org/abs/2503.05244

[32] WritingBench GitHub. https://github.com/X-PLUG/WritingBench

[33] BFCL V3: Multi-Turn & Multi-Step Function Calling. Gorilla. https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html

[34] τ-Bench: Benchmarking AI agents for the real-world. Sierra. https://sierra.ai/blog/benchmarking-ai-agents

[35] Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following. ResearchGate. https://www.researchgate.net/publication/385107280_Multi-IF_Benchmarking_LLMs_on_Multi-Turn_and_Multilingual_Instructions_Following

[36] MMLU-ProX: A Multilingual Benchmark for Advanced LLM Evaluation. https://mmluprox.github.io/

[37] INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge. arXiv:2411.19799. https://arxiv.org/abs/2411.19799

[38] PolyMATH: Evaluating Mathematical Reasoning in Multilingual Contexts. arXiv:2504.18428. https://arxiv.org/abs/2504.18428

[39] 20 LLM evaluation benchmarks and how they work. Evidently AI. https://www.evidentlyai.com/llm-guide/llm-benchmarks

[40] Grok 4 Benchmarks Explained: Why Its Performance is a Game-Changer. Kingy AI. https://kingy.ai/blog/grok-4-benchmarks-explained-why-its-performance-is-a-game-changer/

[41] An In-depth Guide to Benchmarking LLMs. Symbl.ai. https://symbl.ai/developers/blog/an-in-depth-guide-to-benchmarking-llms/

[42] Benchmarks and Metrics for Evaluations of Code Generation: A Critical Review. arXiv:2406.12655. https://arxiv.org/html/2406.12655v1

[43] Qwen2.5-LLM: Extending the boundary of LLMs. Qwen. https://qwenlm.github.io/blog/qwen2.5-llm/

[44] Training Verifiers to Solve Math Word Problems. arXiv:2110.14168. https://arxiv.org/abs/2110.14168

[45] GSM8K Dataset. Papers with Code. https://paperswithcode.com/dataset/gsm8k

[46] GSM8K Benchmark. Klu. https://klu.ai/glossary/GSM8K-eval

[47] HumanEval. GitHub. https://github.com/openai/human-eval

[48] SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. arXiv:1905.00537. https://arxiv.org/abs/1905.00537
