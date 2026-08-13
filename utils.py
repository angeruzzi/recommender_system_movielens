import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# 1. MovieLens metadata
# ============================================================

GENRES = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

MOVIE_COLUMNS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
] + GENRES


# ============================================================
# 2. Data loading
# ============================================================

def download_movielens_100k(base_dir="/content", force=False):
    """
    Baixa e extrai o MovieLens 100K.

    Parameters
    ----------
    base_dir : str, default="/content"
        Diretório onde o dataset será armazenado.

    force : bool, default=False
        Se True, força novo download e extração.

    Returns
    -------
    str
        Caminho para o diretório ml-100k.
    """
    base_dir = os.path.abspath(base_dir)
    dataset_dir = os.path.join(base_dir, "ml-100k")
    zip_path = os.path.join(base_dir, "ml-100k.zip")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    if os.path.isdir(dataset_dir) and not force:
        return dataset_dir

    os.makedirs(base_dir, exist_ok=True)

    if force or not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir)

    return dataset_dir


def load_ratings(data_dir="/content/ml-100k"):
    """
    Carrega o arquivo u.data do MovieLens 100K.

    Returns
    -------
    pandas.DataFrame
        Colunas:
        user_id, item_id, rating, timestamp, datetime
    """
    path = os.path.join(data_dir, "u.data")

    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    ratings["datetime"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s"
    )

    return ratings


def load_movies(data_dir="/content/ml-100k"):
    """
    Carrega o arquivo u.item do MovieLens 100K.

    Returns
    -------
    pandas.DataFrame
        Metadados dos filmes e colunas binárias de gênero.
    """
    path = os.path.join(data_dir, "u.item")

    movies = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        names=MOVIE_COLUMNS
    )

    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        format="%d-%b-%Y",
        errors="coerce"
    )

    return movies


def load_movielens_100k(base_dir="/content", download=True, force=False):
    """
    Carrega ratings e movies do MovieLens 100K.

    Parameters
    ----------
    base_dir : str, default="/content"
        Diretório base.

    download : bool, default=True
        Se True, baixa o dataset caso necessário.

    force : bool, default=False
        Se True, força novo download.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ratings, movies
    """
    data_dir = os.path.join(base_dir, "ml-100k")

    if download:
        data_dir = download_movielens_100k(
            base_dir=base_dir,
            force=force
        )

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    return ratings, movies


# ============================================================
# 3. Temporal split
# ============================================================

def temporal_split_by_user(ratings, train_ratio=0.8):
    """
    Realiza split temporal por usuário.

    Para cada usuário, as interações são ordenadas por datetime.
    As primeiras train_ratio interações vão para treino e as
    restantes para teste.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Deve conter user_id e datetime.

    train_ratio : float, default=0.8
        Proporção de interações de treino por usuário.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        train, test
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio deve estar entre 0 e 1.")

    required_columns = {"user_id", "datetime"}

    if not required_columns.issubset(ratings.columns):
        missing = required_columns - set(ratings.columns)
        raise ValueError(
            f"Colunas obrigatórias ausentes: {sorted(missing)}"
        )

    train_parts = []
    test_parts = []

    for _, user_data in ratings.groupby("user_id"):
        user_data = user_data.sort_values("datetime")

        split_idx = int(len(user_data) * train_ratio)

        # Evita conjuntos vazios quando possível
        split_idx = max(1, split_idx)

        if split_idx >= len(user_data):
            split_idx = len(user_data) - 1

        train_parts.append(user_data.iloc[:split_idx])
        test_parts.append(user_data.iloc[split_idx:])

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    return train, test


def validate_temporal_split(train, test):
    """
    Verifica se, para cada usuário presente em ambos os conjuntos,
    a última interação de treino não ocorre após a primeira de teste.

    Returns
    -------
    pandas.Series
        True/False por usuário.
    """
    train_last = (
        train
        .groupby("user_id")["datetime"]
        .max()
    )

    test_first = (
        test
        .groupby("user_id")["datetime"]
        .min()
    )

    common_users = train_last.index.intersection(
        test_first.index
    )

    return (
        train_last.loc[common_users]
        <= test_first.loc[common_users]
    )


# ============================================================
# 4. Ground truth and candidate items
# ============================================================

def build_ground_truth(
    test,
    relevance_threshold=4,
    train_catalog=None
):
    """
    Constrói o ground truth por usuário.

    Parameters
    ----------
    test : pandas.DataFrame
        Deve conter user_id, item_id e rating.

    relevance_threshold : int or float, default=4
        Rating mínimo para considerar um item relevante.

    train_catalog : set, optional
        Se informado, restringe o ground truth aos itens
        disponíveis no catálogo de treino.

    Returns
    -------
    dict
        {user_id: {item_id, ...}}
    """
    relevant = test[
        test["rating"] >= relevance_threshold
    ]

    ground_truth = (
        relevant
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    if train_catalog is not None:
        ground_truth = {
            user_id: items & train_catalog
            for user_id, items in ground_truth.items()
        }

        ground_truth = {
            user_id: items
            for user_id, items in ground_truth.items()
            if items
        }

    return ground_truth


def build_train_catalog(train):
    """
    Retorna o conjunto de itens presentes no treino.
    """
    return set(train["item_id"].unique())


def build_seen_items(train):
    """
    Constrói os itens já vistos por usuário no conjunto de treino.

    Returns
    -------
    dict
        {user_id: {item_id, ...}}
    """
    return (
        train
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )


def get_candidate_items(user_id, train_catalog, seen_items):
    """
    Retorna itens elegíveis para recomendação.

    Candidates(u) = Catalog_train - Seen_train(u)
    """
    seen = seen_items.get(user_id, set())

    return train_catalog - seen


def build_evaluation_data(
    train,
    test,
    relevance_threshold=4
):
    """
    Constrói os principais objetos necessários para avaliação.

    Returns
    -------
    dict
        Contém:
        - train_catalog
        - seen_items
        - ground_truth
        - evaluable_ground_truth
        - eligible_users
        - evaluation_users
        - new_items_in_test
    """
    train_catalog = build_train_catalog(train)
    seen_items = build_seen_items(train)

    ground_truth = build_ground_truth(
        test,
        relevance_threshold=relevance_threshold
    )

    evaluable_ground_truth = build_ground_truth(
        test,
        relevance_threshold=relevance_threshold,
        train_catalog=train_catalog
    )

    new_items_in_test = (
        set(test["item_id"].unique())
        - train_catalog
    )

    return {
        "train_catalog": train_catalog,
        "seen_items": seen_items,
        "ground_truth": ground_truth,
        "evaluable_ground_truth": evaluable_ground_truth,
        "eligible_users": list(ground_truth.keys()),
        "evaluation_users": list(
            evaluable_ground_truth.keys()
        ),
        "new_items_in_test": new_items_in_test
    }


# ============================================================
# 5. Top-K metrics
# ============================================================

def precision_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    Calcula Precision@K.
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended_items[:k]

    hits = len(
        set(recommended_k)
        & set(relevant_items)
    )

    return hits / k


def recall_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    Calcula Recall@K.
    """
    relevant_items = set(relevant_items)

    if len(relevant_items) == 0:
        return 0.0

    recommended_k = recommended_items[:k]

    hits = len(
        set(recommended_k)
        & relevant_items
    )

    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    Calcula NDCG@K com relevância binária.
    """
    relevant_items = set(relevant_items)

    if len(relevant_items) == 0 or k <= 0:
        return 0.0

    recommended_k = recommended_items[:k]

    dcg = 0.0

    for position, item_id in enumerate(
        recommended_k,
        start=1
    ):
        if item_id in relevant_items:
            dcg += 1 / np.log2(position + 1)

    ideal_hits = min(
        len(relevant_items),
        k
    )

    idcg = sum(
        1 / np.log2(position + 1)
        for position in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# 6. Evaluation pipeline
# ============================================================

def evaluate_recommendations(
    recommendations,
    ground_truth,
    k=10
):
    """
    Avalia recomendações Top-K para múltiplos usuários.

    Parameters
    ----------
    recommendations : dict
        {user_id: [item_1, item_2, ...]}

    ground_truth : dict
        {user_id: {relevant_item_1, ...}}

    k : int, default=10
        Tamanho da lista Top-K.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        results_df, summary
    """
    results = []

    for user_id, relevant_items in ground_truth.items():

        recommended_items = recommendations.get(
            user_id,
            []
        )

        precision = precision_at_k(
            recommended_items,
            relevant_items,
            k
        )

        recall = recall_at_k(
            recommended_items,
            relevant_items,
            k
        )

        ndcg = ndcg_at_k(
            recommended_items,
            relevant_items,
            k
        )

        results.append({
            "user_id": user_id,
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"ndcg@{k}": ndcg
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        summary = {
            f"Precision@{k}": 0.0,
            f"Recall@{k}": 0.0,
            f"NDCG@{k}": 0.0,
            "n_users": 0
        }

        return results_df, summary

    summary = {
        f"Precision@{k}": (
            results_df[f"precision@{k}"].mean()
        ),
        f"Recall@{k}": (
            results_df[f"recall@{k}"].mean()
        ),
        f"NDCG@{k}": (
            results_df[f"ndcg@{k}"].mean()
        ),
        "n_users": len(results_df)
    }

    return results_df, summary


# ============================================================
# 7. Validation
# ============================================================

def validate_metrics():
    """
    Executa testes simples das métricas Top-K.

    Returns
    -------
    bool
        True se todos os testes forem aprovados.
    """

    # Ranking perfeito
    recommended = [1, 2, 3]
    relevant = {1, 2, 3}

    assert precision_at_k(
        recommended,
        relevant,
        k=3
    ) == 1.0

    assert recall_at_k(
        recommended,
        relevant,
        k=3
    ) == 1.0

    assert np.isclose(
        ndcg_at_k(
            recommended,
            relevant,
            k=3
        ),
        1.0
    )

    # Nenhum acerto
    recommended = [1, 2, 3]
    relevant = {4, 5}

    assert precision_at_k(
        recommended,
        relevant,
        k=3
    ) == 0.0

    assert recall_at_k(
        recommended,
        relevant,
        k=3
    ) == 0.0

    assert ndcg_at_k(
        recommended,
        relevant,
        k=3
    ) == 0.0

    # NDCG deve favorecer itens relevantes no topo
    relevant = {10, 20}

    ranking_a = [10, 20, 30, 40]
    ranking_b = [30, 40, 10, 20]

    assert (
        ndcg_at_k(
            ranking_a,
            relevant,
            k=4
        )
        >
        ndcg_at_k(
            ranking_b,
            relevant,
            k=4
        )
    )

    return True


def validate_framework():
    """
    Executa uma validação integrada básica do pipeline de avaliação.

    Returns
    -------
    bool
        True se o framework passar nas validações.
    """
    sample_ground_truth = {
        1: {10, 20},
        2: {30},
        3: {40, 50}
    }

    sample_recommendations = {
        1: [10, 30, 20, 40, 50],
        2: [10, 20, 30, 40, 50],
        3: [40, 60, 70, 80, 90]
    }

    results, summary = evaluate_recommendations(
        recommendations=sample_recommendations,
        ground_truth=sample_ground_truth,
        k=5
    )

    assert len(results) == 3
    assert results["precision@5"].between(0, 1).all()
    assert results["recall@5"].between(0, 1).all()
    assert results["ndcg@5"].between(0, 1).all()
    assert summary["n_users"] == 3

    return True
