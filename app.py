import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── Page Config ──
st.set_page_config(
    page_title="App User Behavior Segmentation",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: bold; color: #0f2027;
        border-left: 5px solid #2c5364;
        padding-left: 10px; margin: 18px 0 10px 0;
    }
    .insight-box {
        background: #f0f7ff; border-radius: 10px;
        padding: 14px 18px; border-left: 5px solid #2c5364;
        margin: 8px 0; color: #1a1a2e;
    }
    .seg-card {
        border-radius: 12px; padding: 16px;
        margin: 6px; color: white; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Process ──
@st.cache_data
def load_data():
    df = pd.read_csv("app_user_behavior_dataset.csv")
    df['rating_given'] = df['rating_given'].fillna(df['rating_given'].median())
    return df

@st.cache_resource
def run_clustering(df):
    features = [
        'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes',
        'feature_clicks_per_session', 'notifications_opened_per_week',
        'in_app_search_count', 'pages_viewed_per_session',
        'days_since_last_login', 'engagement_score', 'churn_risk_score',
        'content_downloads', 'social_shares', 'account_age_days',
        'ads_clicked_last_30_days', 'rating_given'
    ]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    sil = silhouette_score(X_scaled, labels, sample_size=5000)
    db  = davies_bouldin_score(X_scaled, labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Map clusters to segments based on engagement score mean
    df_temp = df.copy()
    df_temp['cluster'] = labels
    eng_mean = df_temp.groupby('cluster')['engagement_score'].mean().sort_values(ascending=False)
    rank_map = {
        eng_mean.index[0]: 'High Engagement',
        eng_mean.index[1]: 'Moderate Engagement',
        eng_mean.index[2]: 'Occasional Users',
        eng_mean.index[3]: 'Low / At-Risk',
    }
    df_temp['segment'] = df_temp['cluster'].map(rank_map)

    return df_temp, X_scaled, X_pca, pca, sil, db, kmeans.inertia_, features

df = load_data()
df_clustered, X_scaled, X_pca, pca, sil, db, inertia, features = run_clustering(df)

seg_colors = {
    'High Engagement':    '#2ecc71',
    'Moderate Engagement':'#3498db',
    'Low / At-Risk':      '#e74c3c',
    'Occasional Users':   '#f39c12'
}
seg_counts = df_clustered['segment'].value_counts()
total = len(df_clustered)

# ── Sidebar ──
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 KPI Dashboard",
    "🔍 EDA & Analysis",
    "🤖 Clustering Model",
    "👥 Cluster Profiles",
    "💡 Business Insights"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"- **Users:** {total:,}")
st.sidebar.markdown(f"- **Features:** {len(features)}")
st.sidebar.markdown(f"- **Clusters:** 4")
st.sidebar.markdown(f"- **Silhouette:** {sil:.4f}")

# ════════════════════════════════
#        PAGE 1: KPI DASHBOARD
# ════════════════════════════════
if page == "🏠 KPI Dashboard":
    st.title("📊 App User Behavior Segmentation")
    st.markdown("**Unsupervised Machine Learning **")
    st.markdown("---")

    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Users</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{sil:.3f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{db:.3f}</div><div class="metric-label">Davies-Bouldin Score</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{pca.explained_variance_ratio_.sum()*100:.1f}%</div><div class="metric-label">PCA Variance Captured</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # KPI Row 2 — Segment cards
    c5, c6, c7, c8 = st.columns(4)
    cols_seg = [c5, c6, c7, c8]
    seg_order = ['High Engagement', 'Moderate Engagement', 'Occasional Users', 'Low / At-Risk']
    card_colors = ['#2ecc71','#3498db','#f39c12','#e74c3c']
    for col, seg, color in zip(cols_seg, seg_order, card_colors):
        cnt = seg_counts.get(seg, 0)
        pct = cnt / total * 100
        col.markdown(f'<div class="metric-card" style="background:{color}"><div class="metric-value">{cnt:,}</div><div class="metric-label">{seg}<br>({pct:.1f}%)</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Segment Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        seg_counts_ordered = seg_counts.reindex(seg_order)
        ax.pie(seg_counts_ordered.values,
               labels=seg_counts_ordered.index,
               colors=card_colors, autopct='%1.1f%%',
               startangle=90, wedgeprops=dict(width=0.55))
        ax.set_title('User Segments')
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Avg Engagement by Segment</div>', unsafe_allow_html=True)
        eng_by_seg = df_clustered.groupby('segment')['engagement_score'].mean().reindex(seg_order)
        fig, ax = plt.subplots(figsize=(5, 4))
        eng_by_seg.plot(kind='bar', ax=ax, color=card_colors, edgecolor='black')
        ax.set_ylabel('Avg Engagement Score')
        ax.set_title('Engagement by Segment')
        ax.tick_params(rotation=30)
        st.pyplot(fig); plt.close()

    with col3:
        st.markdown('<div class="section-header">Avg Churn Risk by Segment</div>', unsafe_allow_html=True)
        churn_by_seg = df_clustered.groupby('segment')['churn_risk_score'].mean().reindex(seg_order)
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_by_seg.plot(kind='bar', ax=ax, color=card_colors, edgecolor='black')
        ax.set_ylabel('Avg Churn Risk Score')
        ax.set_title('Churn Risk by Segment')
        ax.tick_params(rotation=30)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown('<div class="section-header">PCA Cluster Visualization</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for seg, color in seg_colors.items():
            mask = df_clustered['segment'] == seg
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, label=seg, alpha=0.3, s=5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('User Segments in PCA Space')
        ax.legend(markerscale=5, fontsize=8)
        st.pyplot(fig); plt.close()

    with col5:
        st.markdown('<div class="section-header">Sessions per Week by Segment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for seg, color in seg_colors.items():
            subset = df_clustered[df_clustered['segment'] == seg]['sessions_per_week']
            subset.plot(kind='kde', ax=ax, label=seg, color=color, linewidth=2)
        ax.set_title('Sessions/Week Distribution by Segment')
        ax.set_xlabel('Sessions per Week')
        ax.legend(fontsize=8)
        st.pyplot(fig); plt.close()

# ════════════════════════════════
#        PAGE 2: EDA
# ════════════════════════════════
elif page == "🔍 EDA & Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Engagement & Activity", "📱 Demographics", "🔗 Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['engagement_score'].plot(kind='hist', bins=40, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title('Engagement Score Distribution')
            ax.set_xlabel('Engagement Score')
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['churn_risk_score'].plot(kind='hist', bins=40, ax=ax, color='coral', edgecolor='black')
            ax.set_title('Churn Risk Score Distribution')
            ax.set_xlabel('Churn Risk Score')
            st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Subscription Type vs Behavior</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='subscription_type', y='engagement_score', palette='Set2', ax=ax)
            ax.set_title('Subscription vs Engagement')
            st.pyplot(fig); plt.close()
        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='subscription_type', y='churn_risk_score', palette='Set2', ax=ax)
            ax.set_title('Subscription vs Churn Risk')
            st.pyplot(fig); plt.close()

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            df['device_type'].value_counts().plot(kind='bar', ax=ax,
                color=['#3498db','#2ecc71','#e74c3c'], edgecolor='black')
            ax.set_title('Device Type')
            ax.tick_params(rotation=0)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            df['gender'].value_counts().plot(kind='pie', ax=ax,
                autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff','#99ff99'])
            ax.set_title('Gender Distribution')
            ax.set_ylabel('')
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5, 4))
            df['subscription_type'].value_counts().plot(kind='pie', ax=ax,
                autopct='%1.1f%%', startangle=90,
                colors=['#f39c12','#3498db','#2ecc71'])
            ax.set_title('Subscription Type')
            ax.set_ylabel('')
            st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Top Countries</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        df['country'].value_counts().head(8).plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Top 8 Countries by User Count')
        ax.set_ylabel('Users')
        ax.tick_params(rotation=30)
        st.pyplot(fig); plt.close()

    with tab3:
        num_cols = ['sessions_per_week','avg_session_duration_min','daily_active_minutes',
                    'feature_clicks_per_session','notifications_opened_per_week',
                    'days_since_last_login','engagement_score','churn_risk_score',
                    'content_downloads','social_shares','account_age_days']
        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.triu(np.ones((len(num_cols), len(num_cols)), dtype=bool))
        sns.heatmap(df[num_cols].corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════
#        PAGE 3: CLUSTERING
# ════════════════════════════════
elif page == "🤖 Clustering Model":
    st.title("🤖 K-Means Clustering Model")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "K-Means")
    c2.metric("Optimal k", "4")
    c3.metric("Silhouette Score", f"{sil:.4f}")
    c4.metric("Davies-Bouldin", f"{db:.4f}")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📈 Elbow Method", "🔭 PCA Visualization"])

    with tab1:
        st.markdown("**Computing inertia for k = 2 to 10...**")
        inertia_vals = []
        sil_vals = []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia_vals.append(km.inertia_)
            sil_vals.append(silhouette_score(X_scaled, km.labels_, sample_size=3000))

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(list(K_range), inertia_vals, 'bo-', linewidth=2, markersize=8)
            ax.axvline(x=4, color='red', linestyle='--', label='Optimal k=4')
            ax.set_title('Elbow Method — Inertia vs K')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(list(K_range), sil_vals, 'go-', linewidth=2, markersize=8)
            ax.axvline(x=4, color='red', linestyle='--', label='Optimal k=4')
            ax.set_title('Silhouette Score vs K')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.legend()
            st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        for seg, color in seg_colors.items():
            mask = df_clustered['segment'] == seg
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, label=f"{seg} (n={mask.sum():,})",
                       alpha=0.35, s=8)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title(f'PCA — User Cluster Visualization | Total Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%')
        ax.legend(markerscale=4, fontsize=9)
        st.pyplot(fig); plt.close()

# ════════════════════════════════
#        PAGE 4: CLUSTER PROFILES
# ════════════════════════════════
elif page == "👥 Cluster Profiles":
    st.title("👥 Cluster Profiles & User Identification")
    st.markdown("---")

    profile_cols = ['sessions_per_week','avg_session_duration_min','daily_active_minutes',
                    'engagement_score','churn_risk_score','days_since_last_login',
                    'content_downloads','social_shares','feature_clicks_per_session']

    seg_order = ['High Engagement', 'Moderate Engagement', 'Occasional Users', 'Low / At-Risk']
    cluster_profile = df_clustered.groupby('segment')[profile_cols].mean().round(2)

    st.markdown('<div class="section-header">Cluster Profile Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(cluster_profile.T.style.background_gradient(cmap='RdYlGn', axis=1),
                 use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Comparison by Segment</div>', unsafe_allow_html=True)

    selected_feature = st.selectbox("Select a feature to compare:", profile_cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    card_colors = ['#2ecc71','#3498db','#f39c12','#e74c3c']
    cluster_profile.reindex(seg_order)[selected_feature].plot(
        kind='bar', ax=ax, color=card_colors, edgecolor='black')
    ax.set_title(f'{selected_feature.replace("_"," ").title()} by Segment')
    ax.set_ylabel('Average Value')
    ax.tick_params(rotation=25)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Explore Users by Segment</div>', unsafe_allow_html=True)
    selected_seg = st.selectbox("Select Segment:", seg_order)
    seg_df = df_clustered[df_clustered['segment'] == selected_seg][
        ['user_id','age','gender','country','device_type','subscription_type',
         'sessions_per_week','engagement_score','churn_risk_score','days_since_last_login']
    ].reset_index(drop=True)

    st.markdown(f"**{len(seg_df):,} users in this segment** ({len(seg_df)/total*100:.1f}%)")
    st.dataframe(seg_df.head(20), use_container_width=True)

# ════════════════════════════════
#        PAGE 5: INSIGHTS
# ════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Business Insights & Action Mapping")
    st.markdown("---")

    insights = [
        ('High Engagement', '🟢', '#2ecc71',
         f"{seg_counts.get('High Engagement',0):,} users ({seg_counts.get('High Engagement',0)/total*100:.1f}%)",
         'High sessions, long duration, high engagement score, low churn risk',
         ['Offer premium / paid tier upgrades',
          'Enroll in loyalty reward programs',
          'Use as brand ambassadors for referral campaigns',
          'Provide early access to new features']),
        ('Moderate Engagement', '🔵', '#3498db',
         f"{seg_counts.get('Moderate Engagement',0):,} users ({seg_counts.get('Moderate Engagement',0)/total*100:.1f}%)",
         'Average usage, moderate engagement, some churn tendency',
         ['Send personalized push notifications',
          'Highlight features they haven\'t explored',
          'Offer limited-time engagement incentives',
          'Run A/B tests on UI improvements']),
        ('Occasional Users', '🟡', '#f39c12',
         f"{seg_counts.get('Occasional Users',0):,} users ({seg_counts.get('Occasional Users',0)/total*100:.1f}%)",
         'Sporadic logins, low feature interaction, irregular patterns',
         ['Send weekly digest / re-engagement emails',
          'Personalized recommendations based on past behavior',
          'Simplify onboarding flow for returning users',
          'Offer free trial of premium features']),
        ('Low / At-Risk', '🔴', '#e74c3c',
         f"{seg_counts.get('Low / At-Risk',0):,} users ({seg_counts.get('Low / At-Risk',0)/total*100:.1f}%)",
         'Infrequent logins, low activity, high churn risk score',
         ['Immediate win-back campaigns with offers',
          'Survey to understand drop-off reasons',
          'Streamline onboarding and improve UX',
          'Flag for customer support outreach']),
    ]

    for seg, icon, color, count, profile, actions in insights:
        st.markdown(f"""
        <div class="insight-box" style="border-left: 5px solid {color}">
            <b style="font-size:1.1rem">{icon} {seg}</b> &nbsp;|&nbsp; <span style="color:gray">{count}</span><br>
            <b>Profile:</b> {profile}<br>
            <b>Recommended Actions:</b>
            <ul>{''.join(f"<li>{a}</li>" for a in actions)}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Quality Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Silhouette Score", f"{sil:.4f}", "Higher is better ↑")
    col2.metric("Davies-Bouldin", f"{db:.4f}", "Lower is better ↓")
    col3.metric("PCA Variance", f"{pca.explained_variance_ratio_.sum()*100:.1f}%", "2 components")
    col4.metric("Total Segments", "4", "Behavioral clusters")

    st.success("✅ 50,000 users successfully segmented into 4 actionable behavioral groups using K-Means + PCA.")
