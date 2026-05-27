from pathlib import Path
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

DATA_DEMO_DIR = PROJECT_DIR / "data_demo"
DATA_CHINESE_DIR = PROJECT_DIR / "data_chinese_demo"
DATA_CHINESE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DOCS_CSV = DATA_DEMO_DIR / "main_documents.csv"

QUERY_EXPANSION_OUT = DATA_CHINESE_DIR / "query_expansion_zh_extended.csv"
TOPIC_TAXONOMY_OUT = DATA_CHINESE_DIR / "topic_taxonomy_zh.csv"
CHINESE_ANNOTATIONS_OUT = DATA_CHINESE_DIR / "chinese_annotations.csv"


# =========================
# 2. 扩展中文查询词表
# =========================
def build_query_expansion():
    """
    构建扩展版中文查询词表。
    字段：
    query_zh: 中文查询词
    category_zh: 中文类别
    topic_en: 英文主题说明
    expanded_terms: 扩展词，包括英文、德文、古埃及转写等
    explanation_zh: 中文说明
    """
    records = [
        # 神祇与宗教
        ("神", "神祇与宗教", "god / deity", "god,deity,gott,ntr", "nṯr / ntr 是古埃及语中常见的“神”相关词项，也可对应英文 god、deity 和德文 Gott。"),
        ("诸神", "神祇与宗教", "gods / deities", "gods,deities,gott,ntr,ntr.pl", "该主题用于检索复数意义上的神祇、众神或神明群体。"),
        ("女神", "神祇与宗教", "goddess", "goddess,goddesses,deity,ntrt,ntr.t", "该主题用于检索古埃及女神或女性神祇相关文本。"),
        ("大神", "神祇与宗教", "great god", "great,god,ntr,aa,ntr-aa,aA", "“大神”通常对应 great god，也可与古埃及语 ntr-aa / nṯr-ꜥꜣ 相关。"),
        ("太阳神", "神祇与宗教", "sun god / Ra", "sun,solar,ra,re,raw,sonnengott", "Rꜥw / raw 通常与太阳神 Ra / Re 相关。"),
        ("拉神", "神祇与宗教", "Ra / Re", "ra,re,raw,sun,solar", "拉神是古埃及太阳神，常与 ra、re、raw 等词项相关。"),
        ("奥西里斯", "神祇与宗教", "Osiris", "osiris,wsjr", "Wsjr 是奥西里斯 Osiris 的常见古埃及语转写形式。"),
        ("荷鲁斯", "神祇与宗教", "Horus", "horus,hr,heru", "荷鲁斯通常对应 Horus，也常见转写为 hr。"),
        ("赫鲁斯", "神祇与宗教", "Horus", "horus,hr,heru", "赫鲁斯与荷鲁斯同指 Horus，可通过 hr 等词项检索。"),
        ("伊西斯", "神祇与宗教", "Isis", "isis,aset,ast", "伊西斯是奥西里斯神话体系中的重要女神。"),
        ("阿努比斯", "神祇与宗教", "Anubis", "anubis,inpw,anpu", "阿努比斯与木乃伊、墓葬、亡灵保护等主题相关。"),
        ("阿蒙", "神祇与宗教", "Amun", "amun,amon,jmn,imn", "阿蒙是新王国时期重要神祇，常见转写为 jmn / imn。"),
        ("普塔", "神祇与宗教", "Ptah", "ptah,pth", "普塔是孟菲斯地区的重要创造神。"),
        ("托特", "神祇与宗教", "Thoth", "thoth,dhwtj,djehuty", "托特与书写、智慧、审判和神圣记录相关。"),
        ("赛特", "神祇与宗教", "Seth", "seth,set,stx,sty", "赛特常与混乱、敌对、沙漠和神话冲突相关。"),
        ("玛阿特", "神祇与宗教", "Maat", "maat,maa,truth,justice,order", "玛阿特代表真理、秩序、公正与宇宙秩序。"),

        # 王权与政治
        ("国王", "王权与政治", "king / ruler", "king,ruler,nswt,koenig,könig", "nswt 常与古埃及国王或王权相关，也可对应英文 king 和德文 König。"),
        ("法老", "王权与政治", "pharaoh / king", "pharaoh,king,ruler,nswt,pr-aa,per-aa", "法老主题通常与国王、王权和统治者相关。"),
        ("王权", "王权与政治", "kingship / royal power", "kingship,royal,king,ruler,nswt", "王权主题用于检索古埃及国王权力、统治合法性和王室意识形态。"),
        ("王室", "王权与政治", "royal family", "royal,king,queen,prince,princess,nswt", "王室主题涉及国王、王后、王子、公主和王族成员。"),
        ("王后", "王权与政治", "queen", "queen,royal wife,king's wife,hmt-nswt", "王后主题可用于检索国王妻子、王室女性和王后称号。"),
        ("王名", "王权与政治", "royal name", "royal,name,cartouche,nswt", "王名主题涉及国王称号、王名和王室铭文。"),
        ("统治", "王权与政治", "rule / reign", "rule,reign,king,ruler,nswt", "统治主题用于检索国王统治、王朝和政治权力相关文本。"),

        # 来世与死亡
        ("来世", "来世与死亡", "afterlife / underworld", "afterlife,underworld,jenseits,duat,dwat", "来世主题通常与冥界、亡灵、奥西里斯和死后审判相关。"),
        ("冥界", "来世与死亡", "underworld / Duat", "underworld,duat,dwat,jenseits", "冥界主题通常对应 Duat，是古埃及死后世界观的重要概念。"),
        ("死亡", "来世与死亡", "death", "death,dead,tote,mwt", "死亡主题与死者、亡灵、丧葬和来世观念相关。"),
        ("死者", "来世与死亡", "deceased / dead", "dead,deceased,tote,mwt", "死者主题用于检索亡者、受供者和墓葬文本。"),
        ("亡灵", "来世与死亡", "spirit / deceased", "spirit,soul,dead,deceased,ba,ka,mwt", "亡灵主题涉及 ba、ka、死者和来世存在形式。"),
        ("复活", "来世与死亡", "resurrection / rebirth", "resurrection,rebirth,afterlife,osiris,wsjr", "复活主题常与奥西里斯信仰、死后再生和来世观念相关。"),
        ("审判", "来世与死亡", "judgment / tribunal", "judgment,tribunal,court,maat,truth", "审判主题常与亡灵审判、玛阿特和神圣法庭相关。"),
        ("木乃伊", "来世与死亡", "mummy / mummification", "mummy,mummification,embalming,anubis", "木乃伊主题涉及尸体保存、丧葬仪式和阿努比斯信仰。"),
        ("灵魂", "来世与死亡", "soul / ba / ka", "soul,ba,ka,spirit", "灵魂主题可关联 ba、ka 等古埃及生命与死后存在概念。"),

        # 祭祀与供品
        ("供奉", "祭祀与供品", "offering", "offering,opfer,htp", "ḥtp / htp 常与供奉、供品和祭祀公式相关。"),
        ("祭品", "祭祀与供品", "offering", "offering,opfer,htp,bread,beer,incense", "祭品主题涉及供奉给神或死者的物品。"),
        ("供品", "祭祀与供品", "offering goods", "offering,goods,opfer,htp,bread,beer", "供品主题用于检索献给神祇或死者的物品。"),
        ("祭祀", "祭祀与供品", "ritual / cult", "ritual,cult,offering,htp,priest", "祭祀主题涉及宗教仪式、祭司和供奉活动。"),
        ("仪式", "祭祀与供品", "ritual", "ritual,ceremony,cult,offering", "仪式主题用于检索宗教、丧葬或王权仪式相关文本。"),
        ("香", "祭祀与供品", "incense", "incense,sntr,frankincense", "香常作为祭品或宗教仪式中的供奉物。"),
        ("面包", "祭祀与供品", "bread", "bread,t,offering", "面包常作为古埃及供品清单中的重要项目。"),
        ("啤酒", "祭祀与供品", "beer", "beer,hnqt,offering", "啤酒常与供品、宴饮和祭祀清单相关。"),

        # 神庙、墓葬与空间
        ("神庙", "神庙与墓葬", "temple", "temple,tempel,hwt-ntr,pr", "神庙主题涉及神祇崇拜、祭司、仪式和神圣空间。"),
        ("庙宇", "神庙与墓葬", "temple", "temple,tempel,hwt-ntr,pr", "庙宇与神庙同义，常涉及宗教建筑和祭祀活动。"),
        ("墓葬", "神庙与墓葬", "tomb / grave", "tomb,grave,grab,burial", "墓葬主题涉及墓室、墓主、死者和丧葬文本。"),
        ("坟墓", "神庙与墓葬", "tomb / grave", "tomb,grave,grab,burial", "坟墓主题可用于检索墓葬铭文和死者相关文本。"),
        ("铭文", "神庙与墓葬", "inscription", "inscription,text,stela,stele", "铭文主题涉及石碑、墓葬、神庙墙面等文字材料。"),
        ("石碑", "神庙与墓葬", "stela / stele", "stela,stele,inscription", "石碑主题涉及纪念性文本、供奉文本和墓葬材料。"),

        # 文本类型
        ("亡灵书", "文本类型", "Book of the Dead", "book of the dead,totenbuch,tb,afterlife,osiris", "亡灵书是古埃及来世文本的重要类型，常与奥西里斯、冥界和审判相关。"),
        ("金字塔文本", "文本类型", "Pyramid Texts", "pyramid texts,pyramidentexte,pyramid,afterlife", "金字塔文本是古王国王室丧葬文本的重要材料。"),
        ("棺椁文本", "文本类型", "Coffin Texts", "coffin texts,coffin,afterlife", "棺椁文本与中王国丧葬信仰和来世观念相关。"),
        ("医学文本", "文本类型", "medical texts", "medical,medicine,medizin,healing", "医学文本主题涉及疾病、治疗、身体和药物。"),
        ("书信", "文本类型", "letters", "letter,letters,brief,briefe", "书信主题涉及行政、私人交流和日常文本。"),
        ("档案", "文本类型", "archive", "archive,archives,administrative", "档案主题涉及行政记录、经济记录和文书材料。"),
        ("传记", "文本类型", "biography", "biography,autobiography,life", "传记主题涉及人物生平、职务和自我呈现。"),

        # 人物与职业
        ("祭司", "人物与职业", "priest", "priest,wab,hm-ntr,hem-netjer", "祭司主题涉及神庙服务、祭祀活动和宗教职务。"),
        ("书吏", "人物与职业", "scribe", "scribe,schreiber,ss,sesh", "书吏主题涉及书写、文书、行政和知识阶层。"),
        ("官员", "人物与职业", "official", "official,administrator,overseer", "官员主题涉及行政职务、社会身份和墓葬铭文。"),
        ("工匠", "人物与职业", "craftsman", "craftsman,worker,artisan", "工匠主题涉及劳动、技术和社会职业。"),
        ("医生", "人物与职业", "physician / healer", "physician,doctor,healer,medicine", "医生主题与医学文本、治疗和身体知识相关。"),

        # 自然与宇宙
        ("天空", "自然与宇宙", "sky / heaven", "sky,heaven,himmel,pt", "天空主题涉及宇宙结构、神祇和天界观念。"),
        ("大地", "自然与宇宙", "earth / land", "earth,land,ta,tA", "大地主题涉及土地、世界结构和空间观念。"),
        ("尼罗河", "自然与宇宙", "Nile / river", "nile,river,hapy,water", "尼罗河主题涉及河流、泛滥、农业和神圣地理。"),
        ("水", "自然与宇宙", "water", "water,mw,nile,river", "水主题涉及生命、河流、净化和供奉。"),
        ("星辰", "自然与宇宙", "stars", "star,stars,sky,heaven", "星辰主题涉及天空、宇宙和来世升天观念。"),
        ("太阳", "自然与宇宙", "sun", "sun,solar,ra,re,raw", "太阳主题与太阳神、光明、再生和王权相关。"),

        # 抽象概念
        ("生命", "抽象概念", "life", "life,leben,ankh,anh", "生命主题常与 ankh / ꜥnḫ 相关。"),
        ("真理", "抽象概念", "truth / Maat", "truth,maat,maa,justice,order", "真理主题常与玛阿特、秩序和公正相关。"),
        ("秩序", "抽象概念", "order / Maat", "order,maat,truth,justice", "秩序主题涉及宇宙秩序、王权合法性和玛阿特观念。"),
        ("敌人", "抽象概念", "enemy", "enemy,enemies,feind,feinde,hfti", "敌人主题常见于王权、神话斗争和亡灵文本中。"),
        ("胜利", "抽象概念", "victory", "victory,triumph,enemy,enemies", "胜利主题涉及战胜敌人、王权和神话冲突。"),
        ("保护", "抽象概念", "protection", "protection,protect,guardian,amulet", "保护主题常与神祇保护、亡灵保护和护符相关。"),
        ("魔法", "抽象概念", "magic", "magic,spell,heka", "魔法主题涉及咒语、仪式和宗教实践。"),
        ("咒语", "抽象概念", "spell", "spell,magic,utterance,heka", "咒语主题常见于亡灵书、金字塔文本和宗教文本。"),
    ]

    df = pd.DataFrame(
        records,
        columns=[
            "query_zh",
            "category_zh",
            "topic_en",
            "expanded_terms",
            "explanation_zh"
        ]
    )

    df.to_csv(QUERY_EXPANSION_OUT, index=False, encoding="utf-8-sig")
    print(f"扩展中文查询词表已生成：{QUERY_EXPANSION_OUT}")
    print(f"记录数：{len(df)}")
    return df


# =========================
# 3. 中文主题体系
# =========================
def build_topic_taxonomy():
    """
    构建中文主题体系。
    """
    records = [
        ("ZH001", "神祇信仰", "宗教观念", "ntr,god,deity,gott", "涉及古埃及神祇、神明身份、神圣属性和宗教信仰的文本主题。"),
        ("ZH002", "奥西里斯信仰", "神祇信仰", "osiris,wsjr", "涉及奥西里斯、死亡、复活、来世和亡灵审判的文本主题。"),
        ("ZH003", "太阳崇拜", "神祇信仰", "ra,re,raw,sun,solar", "涉及太阳神、太阳运行、光明、再生和太阳神学的文本主题。"),
        ("ZH004", "荷鲁斯信仰", "神祇信仰", "horus,hr", "涉及荷鲁斯、王权保护、神话冲突和王室合法性的文本主题。"),
        ("ZH005", "阿蒙信仰", "神祇信仰", "amun,jmn,imn", "涉及阿蒙神及其神庙、祭司和王权关系的文本主题。"),
        ("ZH006", "王权政治", "社会政治", "king,nswt,ruler,pharaoh", "涉及国王、法老、王权、统治合法性和王室意识形态的文本主题。"),
        ("ZH007", "王室人物", "社会政治", "queen,royal,prince,princess", "涉及王后、王子、公主和王室成员的文本主题。"),
        ("ZH008", "来世观念", "宗教观念", "afterlife,duat,dwat,underworld,jenseits", "涉及冥界、死后世界、亡灵存在和来世旅程的文本主题。"),
        ("ZH009", "死亡与亡灵", "来世观念", "dead,deceased,mwt,soul,ba,ka", "涉及死亡、死者、灵魂、ba、ka 和亡灵身份的文本主题。"),
        ("ZH010", "审判与玛阿特", "来世观念", "judgment,tribunal,maat,truth,justice,order", "涉及亡灵审判、真理、公正、秩序和玛阿特观念的文本主题。"),
        ("ZH011", "祭祀供奉", "宗教实践", "offering,htp,ritual,cult,opfer", "涉及供品、祭祀仪式、神庙供奉和死者供奉的文本主题。"),
        ("ZH012", "神庙空间", "宗教实践", "temple,tempel,hwt-ntr,pr", "涉及神庙、庙宇、祭司和宗教空间的文本主题。"),
        ("ZH013", "墓葬铭文", "文献类型", "tomb,grave,burial,inscription,stela", "涉及墓葬、石碑、死者铭文和墓主身份的文本主题。"),
        ("ZH014", "亡灵文学", "文献类型", "book of the dead,tb,totenbuch,afterlife,osiris", "涉及亡灵书、丧葬咒语和来世通行文本的主题。"),
        ("ZH015", "金字塔文本", "文献类型", "pyramid texts,pyramidentexte,pyramid", "涉及金字塔文本和王室丧葬文学的主题。"),
        ("ZH016", "医学与治疗", "文献类型", "medical,medicine,medizin,healing,physician", "涉及医学文本、治疗、疾病、身体和药物知识的主题。"),
        ("ZH017", "书信与档案", "文献类型", "letter,letters,brief,briefe,archive", "涉及书信、行政档案和文书记录的主题。"),
        ("ZH018", "职业身份", "社会生活", "priest,scribe,official,worker,craftsman", "涉及祭司、书吏、官员、工匠等社会职业身份的主题。"),
        ("ZH019", "自然宇宙", "宇宙观念", "sky,earth,water,nile,sun,star", "涉及天空、大地、尼罗河、水、太阳和星辰等宇宙自然主题。"),
        ("ZH020", "敌人与冲突", "社会政治", "enemy,enemies,hfti,victory,triumph", "涉及敌人、战争、神话冲突和战胜敌人的文本主题。"),
        ("ZH021", "魔法咒语", "宗教实践", "magic,spell,heka,utterance", "涉及魔法、咒语、神圣话语和仪式语言的主题。"),
        ("ZH022", "生命与保护", "宗教观念", "life,ankh,anh,protection,protect", "涉及生命、保护、护佑、重生和神圣庇护的主题。"),
    ]

    df = pd.DataFrame(
        records,
        columns=[
            "topic_id",
            "topic_zh",
            "parent_topic",
            "related_terms",
            "description_zh"
        ]
    )

    df.to_csv(TOPIC_TAXONOMY_OUT, index=False, encoding="utf-8-sig")
    print(f"中文主题体系已生成：{TOPIC_TAXONOMY_OUT}")
    print(f"主题数量：{len(df)}")
    return df


# =========================
# 4. 文档中文主题标注规则
# =========================
TOPIC_RULES = {
    "神祇信仰": ["ntr", "god", "deity", "gott"],
    "奥西里斯信仰": ["osiris", "wsjr"],
    "太阳崇拜": ["ra", "re", "raw", "sun", "solar"],
    "荷鲁斯信仰": ["horus", "hr"],
    "阿蒙信仰": ["amun", "jmn", "imn"],
    "王权政治": ["king", "nswt", "ruler", "pharaoh", "royal"],
    "王室人物": ["queen", "prince", "princess", "royal wife"],
    "来世观念": ["afterlife", "underworld", "duat", "dwat", "jenseits"],
    "死亡与亡灵": ["dead", "deceased", "mwt", "soul", "ba", "ka"],
    "审判与玛阿特": ["judgment", "tribunal", "maat", "truth", "justice", "order"],
    "祭祀供奉": ["offering", "opfer", "htp", "ritual", "cult"],
    "神庙空间": ["temple", "tempel", "hwt-ntr"],
    "墓葬铭文": ["tomb", "grave", "burial", "inscription", "stela", "stele"],
    "亡灵文学": ["book of the dead", "totenbuch", "tb"],
    "金字塔文本": ["pyramid", "pyramidentexte"],
    "医学与治疗": ["medical", "medicine", "medizin", "healing"],
    "书信与档案": ["letter", "letters", "brief", "briefe", "archive"],
    "职业身份": ["priest", "scribe", "official", "worker", "craftsman"],
    "自然宇宙": ["sky", "heaven", "earth", "water", "nile", "star"],
    "敌人与冲突": ["enemy", "enemies", "hfti", "victory", "triumph"],
    "魔法咒语": ["magic", "spell", "heka", "utterance"],
    "生命与保护": ["life", "ankh", "anh", "protection", "protect"],
}


def build_searchable_text(row):
    """
    将一条文档中的多个字段合并为规则匹配文本。
    """
    fields = [
        "translation",
        "transliteration",
        "normalized_transliteration",
        "lemma_forms",
        "mdc",
        "corpus",
        "date",
        "findspot",
    ]

    parts = []
    for field in fields:
        if field in row:
            parts.append(str(row.get(field, "")))

    return " ".join(parts).lower()


def tag_document(row):
    """
    根据规则给文档生成中文主题标签。
    """
    text = build_searchable_text(row)

    matched_topics = []
    matched_rules = []

    for topic, keywords in TOPIC_RULES.items():
        for kw in keywords:
            if kw.lower() in text:
                matched_topics.append(topic)
                matched_rules.append(f"{topic}:{kw}")
                break

    # 去重并保持顺序
    matched_topics = list(dict.fromkeys(matched_topics))
    matched_rules = list(dict.fromkeys(matched_rules))

    return matched_topics, matched_rules


def build_summary_zh(topic_tags, corpus, translation):
    """
    根据主题标签和语料来源生成中文摘要。
    """
    if not topic_tags:
        return "该文本暂未匹配到明确中文主题标签，可结合原始译文、古埃及转写和语料来源进一步判断其内容。"

    topic_str = "、".join(topic_tags)

    base = f"该文本涉及“{topic_str}”等主题"

    if corpus:
        base += f"，来源语料为 {corpus}"

    base += "。"

    if "奥西里斯信仰" in topic_tags:
        base += "文本可能与奥西里斯神话、死亡复活或亡灵信仰相关。"

    if "太阳崇拜" in topic_tags:
        base += "文本可能涉及太阳神、光明、再生或王权神学。"

    if "王权政治" in topic_tags:
        base += "文本可能反映国王、法老、王室身份或统治合法性。"

    if "祭祀供奉" in topic_tags:
        base += "文本可能涉及供品、祭祀仪式或神庙宗教实践。"

    if "来世观念" in topic_tags or "死亡与亡灵" in topic_tags:
        base += "文本可作为古埃及来世观念、亡灵身份或死后世界相关证据。"

    if "审判与玛阿特" in topic_tags:
        base += "文本可能涉及真理、秩序、公正或亡灵审判观念。"

    return base


def build_chinese_annotations():
    """
    为主文档生成中文主题标签与中文摘要。
    """
    print(f"正在读取主文档：{MAIN_DOCS_CSV}")
    df = pd.read_csv(MAIN_DOCS_CSV, dtype=str).fillna("")
    print(f"主文档数量：{len(df)}")

    records = []

    for idx, row in df.iterrows():
        topic_tags, matched_rules = tag_document(row)

        summary_zh = build_summary_zh(
            topic_tags=topic_tags,
            corpus=row.get("corpus", ""),
            translation=row.get("translation", "")
        )

        records.append({
            "doc_id": row.get("doc_id", ""),
            "topic_tags_zh": "；".join(topic_tags),
            "summary_zh": summary_zh,
            "matched_chinese_rules": "；".join(matched_rules),
            "annotation_source": "rule_based"
        })

        if (idx + 1) % 5000 == 0:
            print(f"已处理 {idx + 1} / {len(df)} 条文档")

    ann_df = pd.DataFrame(records)
    ann_df.to_csv(CHINESE_ANNOTATIONS_OUT, index=False, encoding="utf-8-sig")

    print(f"中文文档标注已生成：{CHINESE_ANNOTATIONS_OUT}")
    print(f"标注记录数：{len(ann_df)}")

    print("\n主题标签覆盖情况：")
    exploded = ann_df.copy()
    exploded["topic_tags_zh"] = exploded["topic_tags_zh"].apply(
        lambda x: x.split("；") if isinstance(x, str) and x else []
    )
    exploded = exploded.explode("topic_tags_zh")

    print(exploded["topic_tags_zh"].value_counts().head(30))

    return ann_df


# =========================
# 5. 主流程
# =========================
def main():
    print("==============================")
    print("开始构建中文知识增强层")
    print("==============================")

    query_df = build_query_expansion()
    topic_df = build_topic_taxonomy()
    ann_df = build_chinese_annotations()

    print("\n==============================")
    print("中文知识增强层构建完成！")
    print("==============================")
    print("输出文件：")
    print("1.", QUERY_EXPANSION_OUT)
    print("2.", TOPIC_TAXONOMY_OUT)
    print("3.", CHINESE_ANNOTATIONS_OUT)

    print("\n数据规模：")
    print("扩展中文查询词数量：", len(query_df))
    print("中文主题数量：", len(topic_df))
    print("中文文档标注数量：", len(ann_df))


if __name__ == "__main__":
    main()