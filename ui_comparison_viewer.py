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

prompts = full_data["data"]

# Session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""

# Scroll functions
def scroll_to_top():
    components.html(
        """
        <div id="top"></div>
        <script>
            // Method 1: Standard scroll
            const mainElement = window.parent.document.querySelector('section.main');
            if (mainElement) mainElement.scrollTo(0, 0);
            
            // Method 2: Anchor fallback
            window.location.hash = '#top';
            
            // Method 3: Delayed scroll (double insurance)
            setTimeout(() => {
                if (mainElement) mainElement.scrollTo(0, 0);
                window.parent.scrollTo(0, 0);
            }, 100);
        </script>
        """,
        height=0
    )

# Evaluator input
st.session_state.evaluator_name = st.text_input("👤 Enter your name (for analysis table):", st.session_state.evaluator_name)

st.title("🔢 UI Ranking App (3 Layouts)")

# UI prompt view
if st.session_state.index < len(prompts):
    # Force scroll to top on new prompt
    scroll_to_top()
    time.sleep(0.1)  # Small delay for rendering
    
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
        
        # Triple scroll guarantee
        scroll_to_top()
        components.html(
            """
            <script>
                window.parent.document.querySelector('section.main').scrollTo(0, 0);
                setTimeout(() => {
                    window.scrollTo(0, 0);
                }, 50);
            </script>
            """,
            height=0
        )
        st.markdown('<div id="top"></div>', unsafe_allow_html=True)
        time.sleep(0.2)  # Ensure scroll executes before rerun
        st.rerun()

# Final evaluation
else:
    scroll_to_top()
    time.sleep(0.1)
    
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

    # Metrics section
    st.markdown("### 📈 Evaluation Metrics")
    
    # Mean Rank
    mean_ranks = df[["Claude", "Original Qwen", "New Qwen"]].mean()
    mean_rank_df = mean_ranks.reset_index()
    mean_rank_df.columns = ["Model", "Mean Rank"]
    st.write("**Mean Rank per Model**")
    st.table(mean_rank_df)

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(data=mean_rank_df, x="Model", y="Mean Rank", ax=ax)
    st.pyplot(fig)

# Final scroll insurance
components.html(
    """
    <script>
        if (window.parent.document.readyState === 'complete') {
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        }
    </script>
    """,
    height=0
)




    sns.heatmap(heatmap_data.astype(float), annot=True, cmap="Blues", fmt=".2f", ax=ax3)
    ax3.set_title("Pairwise Win Rate Heatmap")
    st.pyplot(fig3)


