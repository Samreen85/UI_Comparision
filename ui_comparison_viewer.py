import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
input_file = Path("rendered_ui_links.json")
output_file = Path("selected_results.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Load input JSON
with open(input_file, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Use only first 5 prompts
# full_data["data"] = full_data["data"][:5]
prompts = full_data["data"]

# Session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""

# Evaluator input
st.session_state.evaluator_name = st.text_input("👤 Enter your name (for analysis table):", st.session_state.evaluator_name)

st.title("🔢 UI Ranking App (3 Layouts)")

# UI prompt view
if st.session_state.index < len(prompts):
    item = prompts[st.session_state.index]
    st.markdown(f"### Prompt {st.session_state.index + 1} of {len(prompts)}")
    st.markdown(f"**User Prompt:** {item['user_prompt']}")

    layouts = [
        {"label": "Layout 1", "image": item["old"]["ui_link"], "source": "old"},
        {"label": "Layout 2", "image": item["new"]["ui_link"], "source": "new"},
        {"label": "Layout 3", "image": item["new1"]["ui_link"], "source": "new1"}
    ]

    layout_ranks = {}
    for layout in layouts:
        st.markdown(f"#### {layout['label']}")
        st.image(Image.open(layout["image"]), use_container_width=True)
        rank = st.selectbox(
            f"Rank for {layout['label']} (Best = 1)", [1, 2, 3],
            key=f"{layout['label']}_{st.session_state.index}"
        )
        layout_ranks[layout["source"]] = rank

    if st.button("✅ Submit Ranking"):
        ranked_entry = {
            "user_prompt": item["user_prompt"],
            "ranks": layout_ranks
        }
        st.session_state.responses.append(ranked_entry)
        st.session_state.index += 1
        st.rerun()

# Final evaluation
else:
    st.success("✅ All prompts ranked!")

    final_output = {
        "model": full_data.get("model"),
        "prompt_version": full_data.get("prompt_version"),
        "generation_datetime": full_data.get("generation_datetime"),
        "data": []
    }

    rows = []

    for i, entry in enumerate(st.session_state.responses):
        original = prompts[i]
        ranks = entry["ranks"]

        original["old"]["rank"] = ranks.get("old")
        original["new"]["rank"] = ranks.get("new")
        original["new1"]["rank"] = ranks.get("new1")
        final_output["data"].append(original)

        rows.append({
            "Evaluator": st.session_state.evaluator_name,
            "Prompt": entry["user_prompt"],
            "Claude": ranks.get("old"),
            "Original Qwen": ranks.get("new"),
            "New Qwen": ranks.get("new1")
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    st.success(f"📦 JSON results saved to: `{output_file}`")

    # Table (first 5 rows)
    df = pd.DataFrame(rows)
    st.markdown("### 📊 Ranking Table (First 5 Entries Only)")
    st.dataframe(df.head(5))

    # Download CSV
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Table as CSV",
        data=csv_buffer.getvalue(),
        file_name="ui_ranking_results.csv",
        mime="text/csv"
    )

    # === METRICS ===
    st.markdown("### 📈 Evaluation Metrics")

    # Mean Rank with headers
    mean_ranks = df[["Claude", "Original Qwen", "New Qwen"]].mean()
    mean_rank_df = mean_ranks.reset_index()
    mean_rank_df.columns = ["Model", "Mean Rank"]
    st.write("**Mean Rank per Model**")
    st.table(mean_rank_df)

    # Bar chart for Mean Rank
    fig1, ax1 = plt.subplots()
    sns.barplot(data=mean_rank_df, x="Model", y="Mean Rank", ax=ax1, palette="Blues_d")
    ax1.set_title("Mean Rank per Model")
    ax1.set_ylabel("Mean Rank (Lower is Better)")
    st.pyplot(fig1)

    # Normalized Mean Rank with headers
    overall_mean = mean_ranks.mean()
    normalized_mr = mean_ranks / overall_mean * 2
    normalized_mr_df = normalized_mr.reset_index()
    normalized_mr_df.columns = ["Model", "Normalized Rank"]
    st.write("**Normalized Mean Rank**")
    st.table(normalized_mr_df)

    # Bar chart for Normalized Mean Rank
    fig2, ax2 = plt.subplots()
    sns.barplot(data=normalized_mr_df, x="Model", y="Normalized Rank", ax=ax2, palette="Greens_d")
    ax2.set_title("Normalized Mean Rank")
    ax2.set_ylabel("Normalized Score")
    st.pyplot(fig2)

    # Pairwise Win Rates
    st.write("**Pairwise Win Rates**")

    def pairwise_win_rate(df, a, b):
        wins = sum(df[a] < df[b])
        losses = sum(df[a] > df[b])
        return round(wins / (wins + losses), 2) if (wins + losses) > 0 else 0.0

    pairwise = {
        ("Claude", "Original Qwen"): pairwise_win_rate(df, "Claude", "Original Qwen"),
        ("Claude", "New Qwen"): pairwise_win_rate(df, "Claude", "New Qwen"),
        ("Original Qwen", "New Qwen"): pairwise_win_rate(df, "Original Qwen", "New Qwen"),
    }

    win_rate_df = pd.DataFrame([
        {"Model Pair": f"{a} vs {b}", "Win Rate": win_rate}
        for (a, b), win_rate in pairwise.items()
    ])
    st.table(win_rate_df)

    # Heatmap-style Pairwise Win Matrix
    heatmap_data = pd.DataFrame(index=["Claude", "Original Qwen", "New Qwen"], columns=["Claude", "Original Qwen", "New Qwen"], data=1.0)
    for (a, b), win_rate in pairwise.items():
        heatmap_data.loc[a, b] = win_rate
        heatmap_data.loc[b, a] = 1 - win_rate

    fig3, ax3 = plt.subplots()
    sns.heatmap(heatmap_data.astype(float), annot=True, cmap="Blues", fmt=".2f", ax=ax3)
    ax3.set_title("Pairwise Win Rate Heatmap")
    st.pyplot(fig3)

