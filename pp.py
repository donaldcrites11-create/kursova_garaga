import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

DEFAULT_CSV = "network_edges.csv"


def load_edges(csv_path: str) -> pd.DataFrame:
    """Завантажує таблицю каналів зв'язку з CSV-файлу."""
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {csv_path}")

    data = pd.read_csv(path)
    required_columns = {"source", "target", "weight"}

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"У файлі відсутні обов'язкові колонки: {', '.join(missing)}")

    data = data.dropna(subset=["source", "target", "weight"]).copy()
    data["source"] = data["source"].astype(str).str.strip()
    data["target"] = data["target"].astype(str).str.strip()
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    data = data.dropna(subset=["weight"])

    if data.empty:
        raise ValueError("CSV-файл не містить коректних каналів зв'язку.")

    if (data["weight"] < 0).any():
        raise ValueError("Алгоритм Дейкстри працює тільки з невід'ємними вагами каналів.")

    return data


def create_graph(edges: pd.DataFrame) -> nx.Graph:
    """Створює зважений неорієнтований граф за таблицею каналів."""
    graph = nx.Graph()

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            weight=float(row["weight"])
        )

    return graph


def find_optimal_route(graph: nx.Graph, start_node: str, end_node: str) -> tuple[list[str], float]:
    """Знаходить оптимальний маршрут за алгоритмом Дейкстри."""
    if start_node not in graph.nodes:
        raise ValueError(f"Початкового вузла '{start_node}' немає в мережі.")

    if end_node not in graph.nodes:
        raise ValueError(f"Кінцевого вузла '{end_node}' немає в мережі.")

    if start_node == end_node:
        return [start_node], 0.0

    path = nx.dijkstra_path(
        graph,
        source=start_node,
        target=end_node,
        weight="weight"
    )

    weight = nx.dijkstra_path_length(
        graph,
        source=start_node,
        target=end_node,
        weight="weight"
    )

    return path, float(weight)


def draw_network(
    graph: nx.Graph,
    output_file: str,
    route: list[str] | None = None,
    title: str = "Схема мережі"
) -> None:
    """Будує зображення мережі та, за потреби, виділяє оптимальний маршрут."""
    plt.figure(figsize=(10, 7))

    position = nx.spring_layout(graph, seed=7)

    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=1400,
        node_color="#dbeafe",
        edgecolors="#1e3a8a",
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        graph,
        position,
        font_size=12,
        font_weight="bold"
    )

    nx.draw_networkx_edges(
        graph,
        position,
        width=2,
        edge_color="#64748b"
    )

    edge_labels = nx.get_edge_attributes(graph, "weight")
    edge_labels = {edge: int(value) if value == int(value) else value for edge, value in edge_labels.items()}

    nx.draw_networkx_edge_labels(
        graph,
        position,
        edge_labels=edge_labels,
        font_size=10
    )

    if route and len(route) > 1:
        route_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=route_edges,
            width=5,
            edge_color="#dc2626"
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def print_network_table(edges: pd.DataFrame) -> None:
    """Виводить таблицю каналів зв'язку."""
    print("\nТаблиця каналів зв'язку:")
    print(edges.to_string(index=False))


def print_result(start_node: str, end_node: str, path: list[str], weight: float) -> None:
    """Виводить результат пошуку маршруту."""
    print("\nРезультат маршрутизації:")
    print(f"Початковий вузол: {start_node}")
    print(f"Кінцевий вузол: {end_node}")
    print(f"Оптимальний маршрут: {' -> '.join(path)}")
    print(f"Сумарна вага маршруту: {weight:g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пошук оптимального маршруту в мережі з різною вагою каналів зв'язку."
    )

    parser.add_argument(
        "--file",
        default=DEFAULT_CSV,
        help=f"CSV-файл із каналами зв'язку. За замовчуванням: {DEFAULT_CSV}"
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Початковий вузол маршруту"
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Кінцевий вузол маршруту"
    )

    return parser.parse_args()

def main():
    st.title("Пошук оптимального маршруту в мережі")

    # 1. Створюємо поле для завантаження файлу користувачем
    uploaded_file = st.file_uploader("Завантажте власний CSV файл з каналами зв'язку (необов'язково)", type="csv")
    
    # Визначаємо, яке джерело даних використовувати
    if uploaded_file is not None:
        # Якщо користувач завантажив свій файл, беремо його
        data_source = uploaded_file
        st.info("Використовується завантажений вами файл.")
    else:
        # Якщо користувач нічого не завантажив, перевіряємо наявність вбудованого файлу
        if os.path.exists(DEFAULT_CSV):
            data_source = DEFAULT_CSV
            st.success(f"Автоматично завантажено вбудований файл мережі (`{DEFAULT_CSV}`).")
        else:
            st.error(f"Помилка: Вбудований файл `{DEFAULT_CSV}` не знайдено в репозиторії, і жодного файлу не було завантажено.")
            return

    try:
        # Завантажуємо дані з обраного джерела
        edges = load_edges(data_source)
        graph = create_graph(edges)
        
        # Відображаємо таблицю з даними
        st.subheader("Таблиця каналів зв'язку:")
        st.dataframe(edges, use_container_width=True)
        
        # Список доступних вузлів
        nodes = sorted(graph.nodes())
        st.write(f"**Доступні вузли в мережі:** {', '.join(nodes)}")
        
        # Введення точок маршруту
        st.subheader("Параметри маршрутизації")
        start_node = st.text_input("Введіть початковий вузол (наприклад, A):").strip()
        end_node = st.text_input("Введіть кінцевий вузол (наприклад, F):").strip()
        
        if st.button("Розрахувати оптимальний маршрут", type="primary"):
            if start_node and end_node:
                path, weight = find_optimal_route(graph, start_node, end_node)
                
                # Вивід результатів
                st.success(f"🏁 **Оптимальний маршрут:** {' ➡️ '.join(path)}")
                st.info(f"📊 **Сумарна вага (вартість/відстань) маршруту:** {weight:g}")
                
                # Генерація та відображення графіків
                st.subheader("🗺️ Візуалізація мережі")
                
                # Графік 1: Початкова мережа
                draw_network(graph, "temp_initial.png", title="Початкова мережа з вагами каналів")
                st.image("temp_initial.png", caption="Загальна схема каналів зв'язку")
                
                # Графік 2: Виділений маршрут
                draw_network(graph, "temp_route.png", route=path, title=f"Оптимальний маршрут: {' -> '.join(path)}")
                st.image("temp_route.png", caption="Червоним кольором виділено найкоротший шлях")
            else:
                st.warning("⚠️ Будь ласка, вкажіть і початковий, і кінцевий вузли для розрахунку.")
                
    except Exception as error:
        st.error(f"❌ Сталася помилка при обробці даних: {error}")

if __name__ == "__main__":
    main()
