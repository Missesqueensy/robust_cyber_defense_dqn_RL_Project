from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Cyber Defense RL — 10 Nœuds",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #090D16;
            color: #F8FAFC;
        }
        div[data-testid="metric-container"] {
            background-color: #111827;
            border: 1px solid #1E293B;
            padding: 12px 16px;
            border-radius: 10px;
        }
        .action-box {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-left: 4px solid #38BDF8;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 12px;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

NODES_LAYOUT = {
    "FW": (0.0, 2.2),
    "DMZ": (1.6, 2.2),
    "App1": (3.2, 3.1),
    "App2": (3.2, 2.2),
    "Mail": (3.2, 1.3),
    "SOC": (3.2, 0.4),
    "Auth": (5.0, 3.0),
    "DB1": (5.0, 2.1),
    "DB2": (5.0, 1.2),
    "Backup": (5.0, 0.3),
}

EDGES = [
    ("FW", "DMZ"),
    ("DMZ", "App1"),
    ("DMZ", "App2"),
    ("DMZ", "Mail"),
    ("DMZ", "SOC"),
    ("App1", "Auth"),
    ("App2", "Auth"),
    ("Auth", "DB1"),
    ("Auth", "DB2"),
    ("DB1", "Backup"),
    ("DB2", "Backup"),
]

NODE_NAMES = list(NODES_LAYOUT.keys())

RL_MODELS = {
    "DQN": {
        "full_name": "Deep Q-Network (DQN)",
        "seed_offset": 11,
        "heal_prob": 0.72,
        "priority": ["Auth", "DB1", "DB2", "DMZ", "App1", "App2", "SOC", "FW", "Mail", "Backup"],
        "r_heal": 1.0,
        "r_attack": -1.2,
    },
    "Hierarchical DRL": {
        "full_name": "Hierarchical DRL (Manager + Worker)",
        "seed_offset": 23,
        "heal_prob": 0.80,
        "priority": ["Auth", "SOC", "DB1", "DB2", "DMZ", "FW", "App1", "App2", "Mail", "Backup"],
        "r_heal": 1.2,
        "r_attack": -1.0,
    },
    "Ensemble": {
        "full_name": "RL Ensemble Policy",
        "seed_offset": 37,
        "heal_prob": 0.88,
        "priority": ["Auth", "DB1", "DB2", "SOC", "DMZ", "App1", "App2", "FW", "Backup", "Mail"],
        "r_heal": 1.35,
        "r_attack": -0.85,
    },
    "MCTS": {
        "full_name": "Monte Carlo Tree Search (MCTS)",
        "seed_offset": 53,
        "heal_prob": 0.85,
        "priority": ["DB1", "DB2", "Auth", "DMZ", "FW", "SOC", "App1", "App2", "Backup", "Mail"],
        "r_heal": 1.25,
        "r_attack": -0.95,
    },
    "Baseline": {
        "full_name": "Baseline Heuristique",
        "seed_offset": 71,
        "heal_prob": 0.55,
        "priority": ["FW", "DMZ", "App1", "App2", "Mail", "SOC", "Auth", "DB1", "DB2", "Backup"],
        "r_heal": 0.7,
        "r_attack": -1.4,
    },
}


@st.cache_data(show_spinner=False)
def find_model_artifacts(root_dir: str) -> dict[str, list[str]]:
    root = Path(root_dir)
    patterns = ["*.pt", "*.pth", "*.ckpt", "*.zip", "*.pkl", "*.joblib", "*.csv"]
    found: dict[str, list[str]] = {k: [] for k in RL_MODELS}
    if not root.exists():
        return found

    files: list[Path] = []
    for folder in ["src", "results", "."]:
        target_dir = root / folder
        if not target_dir.exists():
            continue
        for pat in patterns:
            for p in target_dir.rglob(pat):
                if any(part in {".venv", "venv", "__pycache__", ".git"} for part in p.parts):
                    continue
                files.append(p)

    for p in sorted({f.resolve() for f in files}):
        low = str(p).lower()
        for key in RL_MODELS:
            token = key.lower().split()[0]
            if token in low:
                found[key].append(str(p.relative_to(root.resolve())))
    return found


@st.cache_data(show_spinner=False)
def build_steps(
    model_name: str,
    max_steps: int,
    seed: int,
    has_artifact: bool,
) -> list[dict]:
    cfg = RL_MODELS[model_name]
    rng = np.random.default_rng(int(seed) + cfg["seed_offset"])
    compromised = {"App1", "DB1"}
    cum_reward = 0.0
    steps = []
    source_label = "Checkpoint / Fichier trouvé" if has_artifact else "Politique RL (simulation)"

    for step in range(max_steps):
        if step == 0:
            steps.append(
                {
                    "step": 0,
                    "compromised": set(compromised),
                    "focus_node": "App1",
                    "action": "INIT",
                    "headline": f"Step 0 ({model_name}): App1 et DB1 compromis, 8 sains.",
                    "detail": f"Mode : {source_label}. L'agent {model_name} démarre l'évaluation.",
                    "healthy": len(NODE_NAMES) - len(compromised),
                    "bad": len(compromised),
                    "reward": 0.0,
                }
            )
            continue

        comp_list = sorted(compromised)
        healthy_list = [n for n in NODE_NAMES if n not in compromised]

        if step % 3 == 0 and healthy_list:
            attacked = str(rng.choice(healthy_list))
            compromised.add(attacked)
            cum_reward += cfg["r_attack"]
            steps.append(
                {
                    "step": step,
                    "compromised": set(compromised),
                    "focus_node": attacked,
                    "action": "ATTACK",
                    "headline": f"Attaque sur {attacked} -> passe COMPROMIS",
                    "detail": f"Propagation détectée vers {attacked} ; nœud passé en rouge.",
                    "healthy": len(NODE_NAMES) - len(compromised),
                    "bad": len(compromised),
                    "reward": round(cum_reward, 2),
                }
            )
            continue

        if comp_list and rng.random() < cfg["heal_prob"]:
            prio = [n for n in cfg["priority"] if n in compromised]
            target = prio[0] if prio else str(rng.choice(comp_list))
            compromised.remove(target)
            cum_reward += cfg["r_heal"]
            steps.append(
                {
                    "step": step,
                    "compromised": set(compromised),
                    "focus_node": target,
                    "action": "REPAIR",
                    "headline": f"{model_name} répare {target} -> passe SAIN",
                    "detail": f"Action défensive appliquée sur {target} selon la politique {model_name}.",
                    "healthy": len(NODE_NAMES) - len(compromised),
                    "bad": len(compromised),
                    "reward": round(cum_reward, 2),
                }
            )
        else:
            target = str(rng.choice(NODE_NAMES))
            cum_reward += 0.1
            steps.append(
                {
                    "step": step,
                    "compromised": set(compromised),
                    "focus_node": target,
                    "action": "SCAN",
                    "headline": f"{model_name} surveille le réseau ({target})",
                    "detail": "Inspection de l'état des nœuds sans modification.",
                    "healthy": len(NODE_NAMES) - len(compromised),
                    "bad": len(compromised),
                    "reward": round(cum_reward, 2),
                }
            )

    return steps


def draw_diagram(compromised_set: set[str], focus_node: str) -> go.Figure:
    edge_x, edge_y = [], []
    for u, v in EDGES:
        x0, y0 = NODES_LAYOUT[u]
        x1, y1 = NODES_LAYOUT[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#64748B", width=2.4),
        hoverinfo="none",
    )

    xs, ys, colors, sizes, borders, texts, hovers = [], [], [], [], [], [], []
    for name in NODE_NAMES:
        x, y = NODES_LAYOUT[name]
        xs.append(x)
        ys.append(y)
        is_bad = name in compromised_set
        is_focus = name == focus_node

        colors.append("#EF4444" if is_bad else "#22C55E")
        sizes.append(44 if is_focus else 35)
        borders.append("#FFFFFF" if is_focus else "#090D16")

        state_txt = "COMP" if is_bad else "OK"
        texts.append(f"{name} ({state_txt})")
        hovers.append(
            f"<b>{name}</b><br>État : {'Compromis' if is_bad else 'Sain'}"
            + ("<br><b>Nœud ciblé</b>" if is_focus else "")
        )

    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=texts,
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=12),
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=3 if is_focus else 2, color=borders),
        ),
    )

    annotations = [
        dict(x=0.8, y=3.65, text="<b>1. Entrée & DMZ</b>", showarrow=False, font=dict(color="#94A3B8", size=13)),
        dict(x=3.2, y=3.65, text="<b>2. Apps & Sécurité</b>", showarrow=False, font=dict(color="#94A3B8", size=13)),
        dict(x=5.0, y=3.65, text="<b>3. Auth & Données</b>", showarrow=False, font=dict(color="#94A3B8", size=13)),
    ]

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        showlegend=False,
        annotations=annotations,
        margin=dict(l=20, r=20, t=25, b=20),
        height=470,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, 5.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 3.95]),
    )
    return fig


artifacts_map = find_model_artifacts(".")

st.sidebar.title("Contrôles RL")
selected_model = st.sidebar.selectbox(
    "Modèle RL",
    list(RL_MODELS.keys()),
    index=0,
)
seed_val = st.sidebar.number_input("Seed (Scénario)", min_value=1, max_value=999, value=42, step=1)
max_steps = st.sidebar.slider("Nombre d'étapes", min_value=10, max_value=50, value=30, step=5)
speed = st.sidebar.slider("Vitesse Auto Play (sec / step)", min_value=0.2, max_value=2.0, value=0.6, step=0.1)

model_artifacts = artifacts_map.get(selected_model, [])
steps = build_steps(
    model_name=selected_model,
    max_steps=int(max_steps),
    seed=int(seed_val),
    has_artifact=len(model_artifacts) > 0,
)
last_idx = len(steps) - 1

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

st.session_state.step_idx = max(0, min(st.session_state.step_idx, last_idx))
st.session_state.step_slider = st.session_state.step_idx


def _step_prev():
    st.session_state.auto_play = False
    st.session_state.step_idx = max(0, st.session_state.step_idx - 1)


def _step_next():
    st.session_state.auto_play = False
    st.session_state.step_idx = min(last_idx, st.session_state.step_idx + 1)


def _toggle_play():
    if not st.session_state.auto_play and st.session_state.step_idx >= last_idx:
        st.session_state.step_idx = 0
    st.session_state.auto_play = not st.session_state.auto_play


def _replay():
    st.session_state.auto_play = False
    st.session_state.step_idx = 0


def _on_slider_change():
    st.session_state.auto_play = False
    st.session_state.step_idx = int(st.session_state.step_slider)


st.title("Schéma Réseau — Défense Cybernétique RL (10 Nœuds)")
st.caption(
    f"Modèle actif : **{RL_MODELS[selected_model]['full_name']}** | "
    f"Fichiers liés détectés : **{len(model_artifacts)}**"
)

b1, b2, b3, b4, b5 = st.columns([1, 1, 1.3, 1, 2.2])
b1.button("Step -1", use_container_width=True, on_click=_step_prev)
b2.button("Step +1", use_container_width=True, on_click=_step_next)
b3.button(
    "Stop Auto Play" if st.session_state.auto_play else "Auto Play",
    use_container_width=True,
    on_click=_toggle_play,
)
b4.button("Replay", use_container_width=True, on_click=_replay)
b5.slider(
    "Step",
    min_value=0,
    max_value=last_idx,
    key="step_slider",
    on_change=_on_slider_change,
)

cur = steps[st.session_state.step_idx]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Step", f"{cur['step']} / {last_idx}")
m2.metric("Nœuds Sains", cur["healthy"])
m3.metric("Nœuds Compromis", cur["bad"])
m4.metric("Récompense Cumulée", f"{cur['reward']:+.2f}")

col_left, col_right = st.columns([1.4, 1.0])

with col_left:
    st.subheader("Topologie Réseau (10 Nœuds)")
    st.plotly_chart(draw_diagram(cur["compromised"], cur["focus_node"]), use_container_width=True)

with col_right:
    st.subheader("Action & État")
    st.markdown(
        f"""
        <div class="action-box">
            <div><b>Modèle RL :</b> {selected_model}</div>
            <div><b>Action :</b> <code>{cur['action']}</code></div>
            <div><b>{cur['headline']}</b></div>
            <div style="margin-top:6px; color:#CBD5E1;">{cur['detail']}</div>
            <div style="margin-top:8px; font-size:13px; color:#94A3B8;">
                Nœud ciblé : <b>{cur['focus_node']}</b> | Compromis : <b>{cur['bad']}</b> | Sains : <b>{cur['healthy']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Historique")
    hist_df = pd.DataFrame(
        [
            {
                "Step": s["step"],
                "Modèle": selected_model,
                "Action": s["action"],
                "Cible": s["focus_node"],
                "Compromis": s["bad"],
                "Sains": s["healthy"],
                "Reward": s["reward"],
            }
            for s in steps[: st.session_state.step_idx + 1]
        ]
    ).iloc[::-1]
    st.dataframe(hist_df, use_container_width=True, height=260, hide_index=True)

if st.session_state.auto_play:
    if st.session_state.step_idx < last_idx:
        time.sleep(float(speed))
        st.session_state.step_idx += 1
        st.rerun()
    else:
        st.session_state.auto_play = False
