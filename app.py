"""
類似値分析アプリ - Streamlit（拡張版）
N×Nの類似値マトリックスを3列テーブルに変換し、
グループ化・代表配列抽出・代表間類似度分析を提供
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from collections import defaultdict

st.set_page_config(
    page_title="類似値分析ツール",
    page_icon="🔗",
    layout="wide"
)

# =============================================================================
# Union-Find（素集合データ構造）クラス
# =============================================================================


class UnionFind:
    """閾値以上の類似度を持つ配列をグループ化するためのデータ構造"""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        """配列xの根（代表）を見つける（経路圧縮付き）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """配列xとyを同じグループに統合"""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

    def get_groups(self):
        """全グループを辞書形式で取得 {根: [メンバーリスト]}"""
        groups = defaultdict(list)
        for element in self.parent:
            groups[self.find(element)].append(element)
        return dict(groups)


# =============================================================================
# データ処理関数
# =============================================================================
def convert_matrix_to_table(df: pd.DataFrame) -> pd.DataFrame:
    """N×Nマトリックス形式のDataFrameを3列に変換（組み合わせ重複を排除）"""
    df = df.set_index(df.columns[0])
    df.index.name = "配列A"

    df_melted = df.reset_index().melt(
        id_vars="配列A",
        var_name="配列B",
        value_name="類似値"
    )
    df_melted["類似値"] = pd.to_numeric(df_melted["類似値"], errors="coerce")

    # 組み合わせとしてソートし、「ペア」として持つ
    df_melted["ペア"] = df_melted.apply(
        lambda r: tuple(sorted([r["配列A"], r["配列B"]])), axis=1
    )
    df_unique = df_melted.drop_duplicates(subset=["ペア"]).drop(
        columns="ペア").reset_index(drop=True)

    return df_unique


def apply_filters(df, person_filter, min_val, max_val, exclude_self):
    """フィルタリング条件を適用"""
    filtered = df.copy()
    filtered = filtered[(filtered["類似値"] >= min_val) &
                        (filtered["類似値"] <= max_val)]

    if person_filter:
        filtered = filtered[
            (filtered["配列A"].isin(person_filter)) |
            (filtered["配列B"].isin(person_filter))
        ]

    if exclude_self:
        filtered = filtered[filtered["配列A"] != filtered["配列B"]]

    return filtered


def get_similarity_value(df_matrix, elem_a, elem_b):
    """マトリックスから2配列間の類似度を取得"""
    try:
        return df_matrix.loc[elem_a, elem_b]
    except KeyError:
        return np.nan


def group_elements_by_threshold(df_table, all_elements, threshold):
    """
    閾値以上の類似度を持つ配列をグループ化

    Parameters:
    - df_table: 3列形式のデータフレーム（配列A, 配列B, 類似値）
    - all_elements: 全配列のリスト
    - threshold: グループ化の閾値

    Returns:
    - groups: {グループID: [メンバーリスト]}
    """
    uf = UnionFind(all_elements)

    # 閾値以上のペアを統合
    high_similarity = df_table[
        (df_table["類似値"] >= threshold) &
        (df_table["配列A"] != df_table["配列B"])
    ]

    for _, row in high_similarity.iterrows():
        uf.union(row["配列A"], row["配列B"])

    # グループを取得し、グループIDを振り直す
    raw_groups = uf.get_groups()
    groups = {f"G{i+1}": sorted(members) for i, members in enumerate(
        sorted(raw_groups.values(), key=lambda x: (-len(x), x[0])))}

    return groups


def select_representative(group_members, df_matrix, method="centroid"):
    """
    グループから代表配列を選択

    Parameters:
    - group_members: グループメンバーのリスト
    - df_matrix: 類似度マトリックス（DataFrame）
    - method: 選択方法 ("centroid", "first", "alphabetical")

    Returns:
    - 代表配列
    """
    if len(group_members) == 1:
        return group_members[0]

    if method == "centroid":
        # グループ内の他メンバーとの平均類似度が最も高い配列
        best_elem = None
        best_avg = -1

        for elem in group_members:
            similarities = []
            for other in group_members:
                if elem != other:
                    sim = get_similarity_value(df_matrix, elem, other)
                    if not np.isnan(sim):
                        similarities.append(sim)

            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim > best_avg:
                    best_avg = avg_sim
                    best_elem = elem

        return best_elem if best_elem else group_members[0]

    elif method == "first":
        return group_members[0]

    elif method == "alphabetical":
        return sorted(group_members)[0]

    return group_members[0]


def create_representative_matrix(representatives, df_matrix):
    """代表配列間の類似度マトリックスを作成"""
    data = []
    for rep_a in representatives:
        row = {"配列": rep_a}
        for rep_b in representatives:
            row[rep_b] = get_similarity_value(df_matrix, rep_a, rep_b)
        data.append(row)

    return pd.DataFrame(data).set_index("配列")


# =============================================================================
# メインアプリ
# =============================================================================
st.title("🔗 類似値分析ツール")

# タブで機能を分割
tab1, tab2 = st.tabs(["📊 基本分析", "🎯 グループ化・代表抽出"])

# サイドバー：共通のデータ入力
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをアップロード", type=["xlsx", "xls"])

if st.sidebar.button("📊 サンプルデータで試す"):
    sample_names = ["A子", "B太", "C美", "D郎", "E子", "F介", "G代", "H男"]
    # 意図的にグループができるようなサンプルデータ
    sample_matrix = [
        [100, 95, 92, 45, 40, 30, 25, 20],  # A子: B太,C美と高類似
        [95, 100, 90, 50, 45, 35, 30, 25],  # B太: A子,C美と高類似
        [92, 90, 100, 48, 42, 32, 28, 22],  # C美: A子,B太と高類似
        [45, 50, 48, 100, 88, 85, 40, 35],  # D郎: E子,F介と高類似
        [40, 45, 42, 88, 100, 90, 38, 33],  # E子: D郎,F介と高類似
        [30, 35, 32, 85, 90, 100, 42, 37],  # F介: D郎,E子と高類似
        [25, 30, 28, 40, 38, 42, 100, 93],  # G代: H男と高類似
        [20, 25, 22, 35, 33, 37, 93, 100],  # H男: G代と高類似
    ]
    sample_data = {"": sample_names}
    for i, name in enumerate(sample_names):
        sample_data[name] = sample_matrix[i]
    st.session_state["sample_df"] = pd.DataFrame(sample_data)
    st.sidebar.success("サンプルデータを読み込みました！")

# データ読み込み
df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file, header=0)
elif "sample_df" in st.session_state:
    df_raw = st.session_state["sample_df"]

# =============================================================================
# タブ1: 基本分析（従来機能）
# =============================================================================
with tab1:
    st.markdown("N×Nの類似値マトリックスを読み込み、組み合わせごとのテーブルを作成します。")

    if df_raw is not None:
        with st.expander("📋 元のマトリックスデータ", expanded=False):
            st.dataframe(df_raw)

        df_table = convert_matrix_to_table(df_raw)
        all_elements = sorted(
            set(df_table["配列A"].tolist() + df_table["配列B"].tolist()))

        # フィルタリング
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            person_filter = st.multiselect(
                "配列で絞り込み", options=all_elements, key="tab1_filter")
        with col2:
            min_val, max_val = st.slider(
                "類似値の範囲", 0, 100, (0, 100), key="tab1_slider")
        with col3:
            exclude_self = st.checkbox("同一配列を除外", key="tab1_exclude")

        # ソート
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_by = st.selectbox(
                "ソート基準", ["類似値", "配列A", "配列B"], key="tab1_sort")
        with sort_col2:
            sort_order = st.radio(
                "順序", ["降順", "昇順"], horizontal=True, key="tab1_order")

        df_filtered = apply_filters(
            df_table, person_filter, min_val, max_val, exclude_self)
        df_sorted = df_filtered.sort_values(by=sort_by, ascending=(
            sort_order == "昇順")).reset_index(drop=True)

        st.subheader(f"📊 結果（{len(df_sorted):,} 件）")
        st.dataframe(df_sorted, use_container_width=True, height=400)

        # 統計
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("件数", f"{len(df_sorted):,}")
        c2.metric("平均", f"{df_sorted['類似値'].mean():.1f}")
        c3.metric("最大", f"{df_sorted['類似値'].max():.0f}")
        c4.metric("最小", f"{df_sorted['類似値'].min():.0f}")

        # CSVダウンロード
        csv = df_sorted.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 CSVダウンロード", csv, "similarity_table.csv", "text/csv", key="tab1_download")
    else:
        st.info("👈 サイドバーからExcelをアップロードしてください")


# =============================================================================
# タブ2: グループ化・代表抽出
# =============================================================================
with tab2:
    st.markdown("""
    ### 🎯 グループ化・代表配列抽出
    
    類似度が高い配列同士をグループ化し、各グループから代表配列を抽出します。
    """)

    if df_raw is not None:
        # マトリックス形式のDataFrameを作成
        df_matrix = df_raw.set_index(df_raw.columns[0])
        df_matrix.index.name = None

        # 3列形式のテーブル
        df_table = convert_matrix_to_table(df_raw)
        all_elements = sorted(
            set(df_table["配列A"].tolist() + df_table["配列B"].tolist()))

        st.markdown("---")

        # 設定パネル
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚙️ グループ化設定")
            threshold = st.select_slider(
                "類似度の閾値（この値以上で同一グループ化）",
                options=list(range(50, 101, 10)),
                value=80,
                help="閾値を高くするとグループが細かく分かれます"
            )

        with col2:
            st.subheader("👤 代表配列の選択方法")
            rep_method = st.radio(
                "選択方法",
                options=["centroid", "alphabetical", "first"],
                format_func=lambda x: {
                    "centroid": "🎯 重心法（グループ内平均類似度が最高）",
                    "alphabetical": "🔤 名前順（アルファベット/50音順で最初）",
                    "first": "📍 登場順（データ内で最初に出現）"
                }[x],
                help="重心法がグループを最もよく代表します"
            )

        st.markdown("---")

        # グループ化実行
        groups = group_elements_by_threshold(df_table, all_elements, threshold)

        # 代表配列を選択
        representatives = {}
        for group_id, members in groups.items():
            rep = select_representative(members, df_matrix, method=rep_method)
            representatives[group_id] = {
                "代表配列": rep,
                "メンバー": members,
                "メンバー数": len(members)
            }

        # グループ化結果を表示
        st.subheader(f"📊 グループ化結果（閾値: {threshold}以上）")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("総配列数", len(all_elements))
            st.metric("グループ数", len(groups))
            st.metric("代表配列数", len(groups))

        with col2:
            # グループ一覧テーブル
            group_data = []
            for group_id, info in representatives.items():
                group_data.append({
                    "グループID": group_id,
                    "代表配列": info["代表配列"],
                    "メンバー数": info["メンバー数"],
                    "メンバー一覧": ", ".join(info["メンバー"])
                })

            df_groups = pd.DataFrame(group_data)
            st.dataframe(df_groups, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 代表配列間の類似度マトリックス
        st.subheader("🔗 代表配列間の類似度マトリックス")

        rep_list = [info["代表配列"] for info in representatives.values()]
        df_rep_matrix = create_representative_matrix(rep_list, df_matrix)

        # ヒートマップ風に色付け
        def highlight_similarity(val):
            if pd.isna(val):
                return ""
            if val == 100:
                return "background-color: #90EE90"  # 緑（自己）
            elif val >= 80:
                return "background-color: #FFB6C1"  # ピンク（高類似）
            elif val >= 60:
                return "background-color: #FFFACD"  # 黄色（中類似）
            else:
                return "background-color: #E0E0E0"  # グレー（低類似）

        styled_matrix = df_rep_matrix.style.applymap(
            highlight_similarity).format("{:.0f}")
        st.dataframe(styled_matrix, use_container_width=True)

        st.caption("🟢 100（自己） | 🔴 80以上（高） | 🟡 60以上（中） | ⚪ 60未満（低）")

        st.markdown("---")

        # 代表配列間の組み合わせテーブル（3列形式）
        st.subheader("📋 代表配列間の類似度一覧")

        rep_pairs = []
        for i, rep_a in enumerate(rep_list):
            for j, rep_b in enumerate(rep_list):
                if i < j:  # 重複排除
                    sim = get_similarity_value(df_matrix, rep_a, rep_b)
                    # どのグループの代表かを取得
                    group_a = [gid for gid, info in representatives.items(
                    ) if info["代表配列"] == rep_a][0]
                    group_b = [gid for gid, info in representatives.items(
                    ) if info["代表配列"] == rep_b][0]
                    rep_pairs.append({
                        "グループA": group_a,
                        "代表A": rep_a,
                        "グループB": group_b,
                        "代表B": rep_b,
                        "類似度": sim
                    })

        df_rep_pairs = pd.DataFrame(rep_pairs).sort_values(
            "類似度", ascending=False).reset_index(drop=True)
        st.dataframe(df_rep_pairs, use_container_width=True, height=300)

        # CSVダウンロード
        st.markdown("---")
        st.subheader("📥 データエクスポート")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_groups = df_groups.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "グループ一覧CSV",
                csv_groups,
                "groups.csv",
                "text/csv",
                key="download_groups"
            )

        with col2:
            csv_matrix = df_rep_matrix.reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "代表間マトリックスCSV",
                csv_matrix,
                "representative_matrix.csv",
                "text/csv",
                key="download_matrix"
            )

        with col3:
            csv_pairs = df_rep_pairs.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "代表間ペアCSV",
                csv_pairs,
                "representative_pairs.csv",
                "text/csv",
                key="download_pairs"
            )

        # 詳細説明
        with st.expander("📖 処理の詳細説明"):
            st.markdown(f"""
            ### グループ化アルゴリズム
            
            **Union-Find（素集合データ構造）** を使用しています。
            
            1. 全配列を個別のグループとして初期化
            2. 類似度が **{threshold}以上** のペアを見つける
            3. 該当するペアの配列を同じグループに統合
            4. 推移的にグループ化（A-B、B-Cが高類似ならA,B,Cは同一グループ）
            
            ### 代表配列の選択方法
            
            - **重心法**: グループ内の他メンバーとの平均類似度が最も高い配列
              - グループの「中心」に位置する配列を選択
              - 最もグループを代表する配列
            - **名前順**: アルファベット/50音順で最初の配列
            - **登場順**: データ内で最初に出現する配列
            
            ### 備考
            
            - グループ内の組み合わせは類似度が高いため検討不要
            - 代表間の類似度が低いペア → 異質なグループ間の関係
            - 代表間の類似度が中程度のペア → 統合を検討すべきグループ
            """)

    else:
        st.info("👈 サイドバーからExcelをアップロードしてください")

        st.markdown("""
        ### 📖 使い方
        
        1. **Excelファイルをアップロード**
           - N×Nの類似度マトリックス形式
        
        2. **閾値を設定**
           - 50〜100の範囲で10刻み
           - 高い閾値 → 厳しい条件（グループが細かく分かれる）
           - 低い閾値 → 緩い条件（大きなグループができる）
        
        3. **代表配列の選択方法を選ぶ**
           - 重心法（推奨）: グループの中心的な配列
        
        4. **結果を確認**
           - グループ一覧
           - 代表配列間の類似度マトリックス
           - 代表配列間のペア一覧
        
        5. **CSVでエクスポート**
           - 各種データをダウンロード可能
        """)
