from pathlib import Path
import sqlite3
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

DATA_DEMO_DIR = PROJECT_DIR / "data_demo"
DATA_CHINESE_DIR = PROJECT_DIR / "data_chinese_demo"
DATABASE_DEMO_DIR = PROJECT_DIR / "database_demo"
DATABASE_DEMO_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DOCS_CSV = DATA_DEMO_DIR / "main_documents.csv"
TERM_DICTIONARY_CSV = DATA_DEMO_DIR / "term_dictionary.csv"
INVERTED_FILE_CSV = DATA_DEMO_DIR / "inverted_file.csv"

QUERY_EXPANSION_ZH_EXTENDED_CSV = DATA_CHINESE_DIR / "query_expansion_zh_extended.csv"
TOPIC_TAXONOMY_ZH_CSV = DATA_CHINESE_DIR / "topic_taxonomy_zh.csv"
CHINESE_ANNOTATIONS_CSV = DATA_CHINESE_DIR / "chinese_annotations.csv"

DB_PATH = DATABASE_DEMO_DIR / "egypt_demo.db"


# =========================
# 2. 安全读取 CSV
# =========================
def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """
    安全读取 CSV 文件。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在：{csv_path}")

    print(f"正在读取：{csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
    print(f"读取完成：{csv_path.name}，行数：{len(df)}，字段数：{len(df.columns)}")

    return df


# =========================
# 3. 创建索引
# =========================
def create_indexes(conn: sqlite3.Connection):
    """
    为 SQLite 表创建索引，提高检索速度。
    """
    cursor = conn.cursor()

    print("\n正在创建索引...")

    index_sql_list = [
        # 主文档表索引
        "CREATE INDEX IF NOT EXISTS idx_main_documents_doc_id ON main_documents(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_main_documents_corpus ON main_documents(corpus);",
        "CREATE INDEX IF NOT EXISTS idx_main_documents_date ON main_documents(date);",

        # 索引词典表索引
        "CREATE INDEX IF NOT EXISTS idx_term_dictionary_term ON term_dictionary(term);",
        "CREATE INDEX IF NOT EXISTS idx_term_dictionary_term_id ON term_dictionary(term_id);",

        # 倒排档索引
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term_id ON inverted_file(term_id);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term ON inverted_file(term);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_doc_id ON inverted_file(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_inverted_file_term_doc ON inverted_file(term_id, doc_id);",

        # 中文查询扩展表索引
        "CREATE INDEX IF NOT EXISTS idx_query_expansion_query_zh ON query_expansion(query_zh);",
        "CREATE INDEX IF NOT EXISTS idx_query_expansion_category_zh ON query_expansion(category_zh);",

        # 中文主题体系索引
        "CREATE INDEX IF NOT EXISTS idx_topic_taxonomy_topic_id ON topic_taxonomy_zh(topic_id);",
        "CREATE INDEX IF NOT EXISTS idx_topic_taxonomy_topic_zh ON topic_taxonomy_zh(topic_zh);",
        "CREATE INDEX IF NOT EXISTS idx_topic_taxonomy_parent_topic ON topic_taxonomy_zh(parent_topic);",

        # 中文文档标注索引
        "CREATE INDEX IF NOT EXISTS idx_chinese_annotations_doc_id ON chinese_annotations(doc_id);",
        "CREATE INDEX IF NOT EXISTS idx_chinese_annotations_topic_tags ON chinese_annotations(topic_tags_zh);",
    ]

    for sql in index_sql_list:
        cursor.execute(sql)

    conn.commit()
    print("索引创建完成。")


# =========================
# 4. 数据库检查
# =========================
def check_database(conn: sqlite3.Connection):
    """
    检查数据库表是否写入成功。
    """
    cursor = conn.cursor()

    print("\n==============================")
    print("数据库检查")
    print("==============================")

    tables = [
        "main_documents",
        "term_dictionary",
        "inverted_file",
        "query_expansion",
        "topic_taxonomy_zh",
        "chinese_annotations"
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} 条记录")

    print("\n检查重要中文查询词：")
    important_queries = [
        "神", "奥西里斯", "太阳神", "国王", "法老",
        "亡灵书", "祭祀", "审判", "复活", "神庙"
    ]

    for query in important_queries:
        cursor.execute(
            """
            SELECT query_zh, category_zh, expanded_terms
            FROM query_expansion
            WHERE query_zh = ?
            LIMIT 1;
            """,
            (query,)
        )

        row = cursor.fetchone()

        if row is None:
            print(f"{query}: 未找到")
        else:
            print(f"{query}: category={row[1]}, expanded_terms={row[2]}")

    print("\n检查重要检索词：")
    important_terms = ["ntr", "wsjr", "nswt", "htp", "king", "god", "osiris", "ra", "raw"]

    for term in important_terms:
        cursor.execute(
            """
            SELECT term_id, term, df, total_tf, fields
            FROM term_dictionary
            WHERE term = ?
            LIMIT 1;
            """,
            (term,)
        )
        row = cursor.fetchone()

        if row is None:
            print(f"{term}: 未找到")
        else:
            print(f"{term}: term_id={row[0]}, df={row[2]}, total_tf={row[3]}, fields={row[4]}")

    print("\n检查中文标注示例：")
    cursor.execute(
        """
        SELECT doc_id, topic_tags_zh, summary_zh
        FROM chinese_annotations
        WHERE topic_tags_zh != ''
        LIMIT 5;
        """
    )
    rows = cursor.fetchall()

    for row in rows:
        print("-" * 80)
        print("doc_id:", row[0])
        print("topic_tags_zh:", row[1])
        print("summary_zh:", row[2])


# =========================
# 5. 主流程
# =========================
def main():
    print("==============================")
    print("开始构建 V1.1 中文知识增强版 SQLite 数据库")
    print("==============================")
    print("数据库路径：", DB_PATH)

    # 读取核心检索数据
    main_df = read_csv_safely(MAIN_DOCS_CSV)
    term_df = read_csv_safely(TERM_DICTIONARY_CSV)
    inverted_df = read_csv_safely(INVERTED_FILE_CSV)

    # 读取中文知识增强层
    query_expansion_df = read_csv_safely(QUERY_EXPANSION_ZH_EXTENDED_CSV)
    topic_taxonomy_df = read_csv_safely(TOPIC_TAXONOMY_ZH_CSV)
    chinese_annotations_df = read_csv_safely(CHINESE_ANNOTATIONS_CSV)

    # 数值字段转换
    if "token_count" in main_df.columns:
        main_df["token_count"] = pd.to_numeric(
            main_df["token_count"],
            errors="coerce"
        ).fillna(0).astype(int)

    for col in ["df", "total_tf", "field_count"]:
        if col in term_df.columns:
            term_df[col] = pd.to_numeric(
                term_df[col],
                errors="coerce"
            ).fillna(0).astype(int)

    if "tf" in inverted_df.columns:
        inverted_df["tf"] = pd.to_numeric(
            inverted_df["tf"],
            errors="coerce"
        ).fillna(0).astype(int)

    # 删除旧数据库
    if DB_PATH.exists():
        print("\n发现旧数据库，正在删除：", DB_PATH)
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    try:
        print("\n正在写入 SQLite 表...")

        main_df.to_sql(
            "main_documents",
            conn,
            if_exists="replace",
            index=False
        )
        print("main_documents 写入完成。")

        term_df.to_sql(
            "term_dictionary",
            conn,
            if_exists="replace",
            index=False
        )
        print("term_dictionary 写入完成。")

        inverted_df.to_sql(
            "inverted_file",
            conn,
            if_exists="replace",
            index=False
        )
        print("inverted_file 写入完成。")

        query_expansion_df.to_sql(
            "query_expansion",
            conn,
            if_exists="replace",
            index=False
        )
        print("query_expansion 写入完成。")

        topic_taxonomy_df.to_sql(
            "topic_taxonomy_zh",
            conn,
            if_exists="replace",
            index=False
        )
        print("topic_taxonomy_zh 写入完成。")

        chinese_annotations_df.to_sql(
            "chinese_annotations",
            conn,
            if_exists="replace",
            index=False
        )
        print("chinese_annotations 写入完成。")

        create_indexes(conn)
        check_database(conn)

        print("\n==============================")
        print("V1.1 中文知识增强版 SQLite 数据库构建完成！")
        print("==============================")
        print("数据库文件：", DB_PATH)

    finally:
        conn.close()

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n数据库大小：{size_mb:.2f} MB")


if __name__ == "__main__":
    main()