
# import streamlit as st
# import json
# from pathlib import Path
# from PIL import Image
# import pandas as pd
# from io import BytesIO
# import matplotlib.pyplot as plt
# import seaborn as sns
# from streamlit_scroll_to_top import scroll_to_here

# # File paths
# input_file = Path("rendered_ui_links.json")
# output_file = Path("selected_results.json")
# output_file.parent.mkdir(parents=True, exist_ok=True)

# # Load input JSON
# with open(input_file, "r", encoding="utf-8") as f:
#     full_data = json.load(f)

# prompts = full_data["data"]

# # Session state
# if "index" not in st.session_state:
#     st.session_state.index = 0
# if "responses" not in st.session_state:
#     st.session_state.responses = []
# if "evaluator_name" not in st.session_state:
#     st.session_state.evaluator_name = ""
# if "scroll" not in st.session_state:
#     st.session_state.scroll = False

# # Scroll to top whenever the flag is set
# if st.session_state.scroll:
#     scroll_to_here(0, key='top')
#     st.session_state.scroll = False

# # Evaluator input
# st.session_state.evaluator_name = st.text_input("👤 Enter your name (for analysis table):", st.session_state.evaluator_name)

# st.title("🔢 UI Ranking App (3 Layouts)")

# def advance():
#     ranked_entry = {
#         "user_prompt": prompts[st.session_state.index]["user_prompt"],
#         "ranks": {
#             "old": st.session_state[f"Layout 1_{st.session_state.index}"],
#             "new": st.session_state[f"Layout 2_{st.session_state.index}"],
#             "new1": st.session_state[f"Layout 3_{st.session_state.index}"]
#         }
#     }
#     st.session_state.responses.append(ranked_entry)
#     st.session_state.index += 1
#     st.session_state.scroll = True

# # UI prompt view
# if st.session_state.index < len(prompts):
#     item = prompts[st.session_state.index]
#     st.markdown(f"### Prompt {st.session_state.index + 1} of {len(prompts)}")
#     st.markdown(f"**User Prompt:** {item['user_prompt']}")

#     layouts = [
#         {"label": "Layout 1", "image": item["old"]["ui_link"], "source": "old"},
#         {"label": "Layout 2", "image": item["new"]["ui_link"], "source": "new"},
#         {"label": "Layout 3", "image": item["new1"]["ui_link"], "source": "new1"}
#     ]

#     for layout in layouts:
#         st.markdown(f"#### {layout['label']}")
#         st.image(Image.open(layout["image"]), use_container_width=True)
#         rank = st.selectbox(
#             f"Rank for {layout['label']} (Best = 1)", [1, 2, 3],
#             key=f"{layout['label']}_{st.session_state.index}"
#         )

#     st.button("✅ Submit Ranking", on_click=advance)

# # Final evaluation
# else:
#     st.success("✅ All prompts ranked!")
#     final_output = {
#         "model": full_data.get("model"),
#         "prompt_version": full_data.get("prompt_version"),
#         "generation_datetime": full_data.get("generation_datetime"),
#         "data": []
#     }

#     rows = []
#     for i, entry in enumerate(st.session_state.responses):
#         original = prompts[i]
#         ranks = entry["ranks"]

#         original["old"]["rank"] = ranks.get("old")
#         original["new"]["rank"] = ranks.get("new")
#         original["new1"]["rank"] = ranks.get("new1")
#         final_output["data"].append(original)

#         rows.append({
#             "Evaluator": st.session_state.evaluator_name,
#             "Prompt": entry["user_prompt"],
#             "Claude Sonnet 4 (old prompt)": ranks.get("old"),
#             "Claude Sonnet 4 (new prompt)": ranks.get("new"),
#             "Claude Opus 4 (new prompt)": ranks.get("new1")
#         })

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)

#     st.success(f"📦 JSON results saved to: `{output_file}`")

#     # Table (first 5 rows)
#     df = pd.DataFrame(rows)
#     st.markdown("### 📊 Ranking Table (First 5 Entries Only)")
#     st.dataframe(df.head(5))

#     # Download CSV
#     csv_buffer = BytesIO()
#     df.to_csv(csv_buffer, index=False)
#     st.download_button(
#         label="⬇️ Download Table as CSV",
#         data=csv_buffer.getvalue(),
#         file_name="ui_ranking_results.csv",
#         mime="text/csv"
#     )

#     # Metrics section
#     st.markdown("### 📈 Evaluation Metrics")

#     df = df.rename(columns={
#     "Claude Sonnet 4 (old prompt)": "Claude Sonnet4 old",
#     "Claude Sonnet 4 (new prompt)": "Claude Sonnet4 new",
#     "Claude Opus 4 (new prompt)": "Claude Opus4 new"
#     })
    
#     # Mean Rank
#     mean_ranks = df[["Claude Sonnet4 old", "Claude Sonnet4 new", "Claude Opus4 new"]].mean()
#     # mean_ranks = df[["Claude Sonnet4 old", "Claude Sonnet4 new", "Claude Opus4 new"]].mean()
#     mean_rank_df = mean_ranks.reset_index()
#     mean_rank_df.columns = ["Model", "Mean Rank"]
#     st.write("**Mean Rank per Model**")
#     st.table(mean_rank_df)

#     # Visualization
#     fig, ax = plt.subplots()
#     sns.barplot(data=mean_rank_df, x="Model", y="Mean Rank", ax=ax)
#     st.pyplot(fig)




import psycopg2
import streamlit as st
from psycopg2.extras import RealDictCursor
from PIL import Image
import requests
from io import BytesIO
from streamlit_scroll_to_top import scroll_to_here

# ——— DB CONFIG ———
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "dc-db-dev"
DB_USER = "postgres"
DB_PASSWORD = "@Mf8hi*kO3iK9Q^q"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD
    )

# ——— FETCH COMPARISONS ———
def fetch_comparisons():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT c.comparisons_id AS comparison_id,
               p.prompt_text,
               r1.ui_screenshot_link AS layout1,
               r2.ui_screenshot_link AS layout2,
               r3.ui_screenshot_link AS layout3
        FROM comparisons c
        JOIN prompts p ON p.prompt_id = c.prompt_id
        JOIN renders r1 ON r1.renders_id = c.render_id_1
        JOIN renders r2 ON r2.renders_id = c.render_id_2
        JOIN renders r3 ON r3.renders_id = c.render_id_3
        WHERE c.output_id_1 IS NOT NULL
          AND c.output_id_2 IS NOT NULL
          AND c.output_id_3 IS NOT NULL
          AND c.render_id_1 IS NOT NULL
          AND c.render_id_2 IS NOT NULL
          AND c.render_id_3 IS NOT NULL
          AND p.prompt_id IS NOT NULL
          AND r1.ui_screenshot_link IS NOT NULL
          AND r2.ui_screenshot_link IS NOT NULL
          AND r3.ui_screenshot_link IS NOT NULL
        ORDER BY c.comparisons_id;
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

# ——— STREAMLIT UI ———
st.set_page_config(page_title="UI Ranking (Test Mode)", layout="centered")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "comparisons" not in st.session_state:
    st.session_state.comparisons = fetch_comparisons()
if "scroll" not in st.session_state:
    st.session_state.scroll = False

if st.session_state.scroll:
    scroll_to_here(0, key="top")
    st.session_state.scroll = False

# Determine max width based on device
comp = None
if st.session_state.index < len(st.session_state.comparisons):
    comp = st.session_state.comparisons[st.session_state.index]

# Default max-width for desktop
max_width = "800px"

# If the current prompt targets mobile, set width to 390px
if comp and "mobile" in comp['prompt_text'].lower():  # adjust field if you have a dedicated device column
    max_width = "390px"

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: {max_width};
        margin: auto;
        padding: 10px;
    }}
    input[type="text"] {{
        max-width: 100% !important;
        width: 100% !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Evaluator name input (not saved during test mode)
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""
st.session_state.evaluator_name = st.text_input(
    "👤 Enter your name:", st.session_state.evaluator_name
)

# Display comparisons
if st.session_state.index < len(st.session_state.comparisons):
    comp = st.session_state.comparisons[st.session_state.index]
    st.markdown(
        f"### Prompt {st.session_state.index + 1} of {len(st.session_state.comparisons)}"
    )
    st.markdown(f"**Prompt:** {comp['prompt_text']}")

    ranks = {}
    for i in range(1, 4):
        layout = f"Layout {i}"
        img_url = comp[f"layout{i}"]
        st.markdown(f"#### {layout}")
        try:
            img_data = requests.get(img_url, timeout=10).content
            img = Image.open(BytesIO(img_data))
            st.image(img, use_container_width=True)
        except:
            st.warning(f"Could not load image for {layout}.")
        ranks[layout] = st.selectbox(
            f"Rank for {layout} (Best = 1)", [1, 2, 3],
            key=f"{layout}_{st.session_state.index}"
        )

    if st.button("Submit Ranking"):
        if not st.session_state.evaluator_name.strip():
            st.error("Please enter your name before proceeding.")
        else:
            st.session_state.responses = getattr(st.session_state, "responses", [])
            st.session_state.responses.append({
                "comparison_id": comp["comparison_id"],
                "ranks": ranks
            })
            st.session_state.index += 1
            st.session_state.scroll = True
            st.rerun()

else:
    st.success("All prompts displayed!")
    st.markdown("**In test mode: no data was saved.**")
    if "responses" in st.session_state:
        st.write("Collected Rankings:", st.session_state.responses)
    else:
        st.write("No rankings collected yet.")




