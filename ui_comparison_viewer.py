
import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_scroll_to_top import scroll_to_here

# File paths
input_file = Path("rendered_ui_links.json")
output_file = Path("selected_results.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Load input JSON
with open(input_file, "r", encoding="utf-8") as f:
    full_data = json.load(f)

prompts = full_data["data"]

# Session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""
if "scroll" not in st.session_state:
    st.session_state.scroll = False

# Scroll to top whenever the flag is set
if st.session_state.scroll:
    scroll_to_here(0, key='top')
    st.session_state.scroll = False

# Evaluator input
st.session_state.evaluator_name = st.text_input("👤 Enter your name (for analysis table):", st.session_state.evaluator_name)

st.title("🔢 UI Ranking App (3 Layouts)")

def advance():
    ranked_entry = {
        "user_prompt": prompts[st.session_state.index]["user_prompt"],
        "ranks": {
            "old": st.session_state[f"Layout 1_{st.session_state.index}"],
            "new": st.session_state[f"Layout 2_{st.session_state.index}"],
            "new1": st.session_state[f"Layout 3_{st.session_state.index}"]
        }
    }
    st.session_state.responses.append(ranked_entry)
    st.session_state.index += 1
    st.session_state.scroll = True

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

    for layout in layouts:
        st.markdown(f"#### {layout['label']}")
        st.image(Image.open(layout["image"]), use_container_width=True)
        rank = st.selectbox(
            f"Rank for {layout['label']} (Best = 1)", [1, 2, 3],
            key=f"{layout['label']}_{st.session_state.index}"
        )

    st.button("✅ Submit Ranking", on_click=advance)

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
            "Claude Sonnet 4 (old prompt)": ranks.get("old"),
            "Claude Sonnet 4 (new prompt)": ranks.get("new"),
            "Claude Opus 4 (new prompt)": ranks.get("new1")
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

    # Metrics section
    st.markdown("### 📈 Evaluation Metrics")

    df = df.rename(columns={
    "Claude Sonnet 4 (old prompt)": "Claude Sonnet4 old",
    "Claude Sonnet 4 (new prompt)": "Claude Sonnet4 new",
    "Claude Opus 4 (new prompt)": "Claude Opus4 new"
    })
    
    # Mean Rank
    mean_ranks = df[["Claude Sonnet4 old", "Claude Sonnet4 new", "Claude Opus4 new"]].mean()
    # mean_ranks = df[["Claude Sonnet4 old", "Claude Sonnet4 new", "Claude Opus4 new"]].mean()
    mean_rank_df = mean_ranks.reset_index()
    mean_rank_df.columns = ["Model", "Mean Rank"]
    st.write("**Mean Rank per Model**")
    st.table(mean_rank_df)

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(data=mean_rank_df, x="Model", y="Mean Rank", ax=ax)
    st.pyplot(fig)



