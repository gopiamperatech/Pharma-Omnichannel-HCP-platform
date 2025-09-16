
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

st.set_page_config(page_title='Omnichannel HCP Engagement Platform (Demo)', layout='wide')

GNE_SPECIALTIES = {
    "All": [],
    "Oncology": ["Oncology"],
    "Immunology": ["Immunology"],
    "Ophthalmology": ["Ophthalmology"],
    "Neuroscience": ["Neurology"],
    "Hematology": ["Hematology","Hemophilia"]
}

@st.cache_data
def load_data():
    hcps = pd.read_csv('data/hcps.csv')
    interactions = pd.read_csv('data/interactions.csv')
    analytics = pd.read_csv('data/analytics.csv')
    assets = pd.read_csv('data/content_assets.csv')
    touches = pd.read_csv('data/touches.csv')
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    touches['date'] = pd.to_datetime(touches['date'])
    return hcps, interactions, analytics, assets, touches

hcps, interactions, analytics, assets, touches = load_data()

st.sidebar.title("Omnichannel HCP Demo (Genentech Specialties)")
gne_spec = st.sidebar.selectbox("Genentech Specialty Preset", list(GNE_SPECIALTIES.keys()))
st.sidebar.caption("Applies specialty-aligned therapy filters across dashboards.")

def apply_gne_filter(df, therapy_col='therapy_area'):
    if gne_spec == "All":
        return df
    allowed = set(GNE_SPECIALTIES[gne_spec])
    if not allowed or therapy_col not in df.columns:
        return df
    return df[df[therapy_col].isin(allowed)]

page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "👩‍⚕️ HCP Profiles",
    "🛣️ Journey Timeline",
    "📈 Engagement Analytics",
    "🏷️ Brand & Therapy (ROI overlay)",
    "🧑‍🤝‍🧑 Rep Scorecards (ROI overlay)",
    "📚 Content Factory",
    "🧭 Next Best Action (rules + scoring)",
    "🧪 Brand/Therapy Filters + Cohorts",
    "💹 Attribution & ROI (global)",
])

st.sidebar.markdown("---")
st.sidebar.caption("Mock demo • Sept 2025")

def attribute_conversions(df, model="Last Touch", lookback_days=21):
    df = df.sort_values(['hcp_id','date'])
    channels = sorted(df['channel'].unique().tolist())
    credit = {ch: 0.0 for ch in channels}
    spend = {ch: 0.0 for ch in channels}
    for ch in channels:
        spend[ch] = float(df[df['channel']==ch]['cost'].sum())
    lambda_decay = 0.1
    for hid in df['hcp_id'].unique():
        sub = df[df['hcp_id']==hid]
        for i, row in sub.iterrows():
            if row['conversion'] == 1:
                conv_date = row['date']
                prior = sub[(sub['date']<=conv_date) & (sub['date']>=conv_date - pd.Timedelta(days=lookback_days))]
                prior = prior[prior['conversion']==0]
                if prior.empty:
                    continue
                if model == "Last Touch":
                    last_ch = prior.sort_values('date').iloc[-1]['channel']
                    credit[last_ch] += 1.0
                elif model == "First Touch":
                    first_ch = prior.sort_values('date').iloc[0]['channel']
                    credit[first_ch] += 1.0
                elif model == "Linear":
                    w = 1.0 / len(prior)
                    for ch in prior['channel'].tolist():
                        credit[ch] += w
                else:  # Time Decay
                    weights = []
                    for j, prow in prior.iterrows():
                        age = (conv_date - prow['date']).days
                        weights.append(np.exp(-lambda_decay * age))
                    total_w = sum(weights)
                    for (j, prow), w in zip(prior.iterrows(), weights):
                        credit[prow['channel']] += w / total_w
    return credit, spend

def compute_roi_table(df, revenue_per_conv, model="Last Touch", lookback_days=21):
    credit, spend = attribute_conversions(df, model=model, lookback_days=lookback_days)
    channels = sorted(set(list(credit.keys()) + list(spend.keys())))
    out = pd.DataFrame({
        'Channel': channels,
        'Attributed Conversions': [credit.get(c,0.0) for c in channels],
        'Spend': [spend.get(c,0.0) for c in channels],
    })
    out['Revenue'] = out['Attributed Conversions'] * revenue_per_conv
    out['ROI'] = np.where(out['Spend']>0, (out['Revenue'] - out['Spend']) / out['Spend'], 0.0)
    return out

def days_since_last(inter_df, channel, ref_time=None):
    if ref_time is None:
        ref_time = inter_df['timestamp'].max() if not inter_df.empty else pd.Timestamp.now()
    c = inter_df[inter_df['channel']==channel]
    if c.empty:
        return 999
    last = c['timestamp'].max()
    return max(0, int((ref_time - last).days))

# ----- Pages -----
if page == "🏠 Home":
    st.title("Omnichannel HCP Engagement Platform — Demo")
    st.write(f"**Genentech Specialty Filter:** {gne_spec}")
    st.subheader("Highlights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Reach %", f"{int(analytics[analytics['metric']=='Reach %']['value'].iloc[0])}%")
    with col2:
        st.metric("Engagement Quality", f"{int(analytics[analytics['metric']=='Engagement Quality %']['value'].iloc[0])}%")
    with col3:
        st.metric("Cost per Engagement", f"${float(analytics[analytics['metric']=='Cost per Engagement']['value'].iloc[0]):.2f}")
    with col4:
        st.metric("Prescribing Uplift", f"{int(analytics[analytics['metric']=='Prescribing Uplift %']['value'].iloc[0])}%")
    st.markdown("---")
    st.write("**Recent Interactions (last 10)**")
    recent = interactions.sort_values('timestamp', ascending=False).head(10).copy()
    st.dataframe(recent[['hcp_id','timestamp','channel','event','details']])

elif page == "👩‍⚕️ HCP Profiles":
    st.title("HCP Profiles")
    st.dataframe(hcps[['hcp_id','name','specialty','location','segments','prescribing_volume','preferred_channels','assigned_rep','last_engagement_date']])

elif page == "🛣️ Journey Timeline":
    st.title("HCP Journey Timeline")
    sel_hcp = st.selectbox("Select HCP", hcps['name'].tolist())
    hcp_row = hcps[hcps['name']==sel_hcp].iloc[0]
    hcp_id = hcp_row['hcp_id']
    j = interactions[interactions['hcp_id']==hcp_id].sort_values('timestamp')
    st.write(f"Showing journey for **{sel_hcp}** ({hcp_id}) — Rep: **{hcp_row['assigned_rep']}**")
    st.dataframe(j[['timestamp','channel','event','details']])

    st.subheader("Channel Events over Time")
    daily = j.groupby(j['timestamp'].dt.date).size().reset_index(name='events')
    if not daily.empty:
        fig, ax = plt.subplots()
        ax.plot(daily['timestamp'], daily['events'], marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Events")
        ax.set_title("Daily Events")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No interactions for this HCP yet.")

    st.markdown("---")
    st.subheader("Export PDF Journey Summary")
    if st.button("Generate PDF for this HCP"):
        buf = BytesIO()
        with PdfPages(buf) as pdf:
            fig1, ax1 = plt.subplots(figsize=(8.27, 11.69))
            ax1.axis('off')
            lines = [
                f"HCP Journey Summary — {sel_hcp} ({hcp_id})",
                f"Specialty: {hcp_row['specialty']} | Location: {hcp_row['location']}",
                f"Segments: {hcp_row['segments']}",
                f"Prescribing Volume: {hcp_row['prescribing_volume']} | Preferred Channels: {hcp_row['preferred_channels']}",
                f"Assigned Rep: {hcp_row['assigned_rep']}",
                "", "Recent Interactions:"
            ]
            y = 0.95
            for ln in lines:
                ax1.text(0.05, y, ln, transform=ax1.transAxes, fontsize=12, va='top'); y -= 0.04
            sub = j.tail(10)[['timestamp','channel','event','details']]
            for _, row in sub.iterrows():
                ax1.text(0.05, y, f"{row['timestamp'].date()} | {row['channel']} | {row['event']} | {row['details']}", transform=ax1.transAxes, fontsize=10, va='top'); y -= 0.03
            pdf.savefig(fig1); plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
            if not daily.empty:
                ax2.plot(daily['timestamp'], daily['events'], marker='o')
                ax2.set_title(f"Daily Events — {sel_hcp}"); ax2.set_xlabel("Date"); ax2.set_ylabel("Events"); plt.xticks(rotation=45)
            else:
                ax2.text(0.5, 0.5, "No interaction data", ha='center', va='center')
            pdf.savefig(fig2); plt.close(fig2)

        st.download_button("Download PDF", buf.getvalue(), file_name=f"{hcp_id}_{sel_hcp.replace(' ','_')}_journey.pdf", mime="application/pdf")

elif page == "📈 Engagement Analytics":
    st.title("Engagement Analytics")
    pref_counts = hcps['preferred_channels'].str.split('|').explode().value_counts().reset_index()
    pref_counts.columns = ['channel','count']
    inter_counts = interactions['channel'].value_counts().reset_index()
    inter_counts.columns = ['channel','count']

    st.subheader("Preferred Channels (All HCPs)")
    st.dataframe(pref_counts)
    fig1, ax1 = plt.subplots()
    ax1.bar(pref_counts['channel'], pref_counts['count'])
    ax1.set_title("Preferred Channels (Profiles)")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig1)

    st.subheader("Interactions by Channel")
    st.dataframe(inter_counts)
    fig2, ax2 = plt.subplots()
    ax2.bar(inter_counts['channel'], inter_counts['count'])
    ax2.set_title("Interactions by Channel")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig2)

elif page == "🏷️ Brand & Therapy (ROI overlay)":
    st.title("Brand & Therapy Dashboard — ROI Overlay")
    t = apply_gne_filter(touches.copy(), therapy_col='therapy_area')
    col1, col2, col3 = st.columns(3)
    with col1:
        brands = ["All"] + sorted(t['brand'].unique().tolist())
        sel_brand = st.selectbox("Brand", brands)
    with col2:
        therapies = ["All"] + sorted(t['therapy_area'].unique().tolist())
        sel_therapy = st.selectbox("Therapy Area", therapies)
    with col3:
        model = st.selectbox("Attribution Model", ["Last Touch","First Touch","Linear","Time Decay"])

    revenue_per_conv = st.slider("Revenue per conversion (mock)", 100, 2000, 500, step=50)
    lookback_days = st.slider("Lookback (days)", 7, 45, 21)

    df = t.copy()
    if sel_brand != "All": df = df[df['brand']==sel_brand]
    if sel_therapy != "All": df = df[df['therapy_area']==sel_therapy]

    if df.empty:
        st.info("No touches for current filters.")
    else:
        st.subheader("Spend/Touches/Conversions")
        grp = df.groupby('channel').agg(spend=('cost','sum'),
                                        touches_cnt=('channel','count'),
                                        conversions=('conversion','sum')).reset_index()
        grp['CPC'] = np.where(grp['touches_cnt']>0, grp['spend']/grp['touches_cnt'], 0.0)
        st.dataframe(grp)

        st.subheader("Attribution-based ROI (overlay)")
        roi_tbl = compute_roi_table(df, revenue_per_conv, model=model, lookback_days=lookback_days)
        st.dataframe(roi_tbl)

        figA, axA = plt.subplots()
        axA.bar(roi_tbl['Channel'], roi_tbl['ROI'])
        axA.set_title(f"ROI by Channel (Model: {model})")
        plt.setp(axA.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(figA)

        figB, axB = plt.subplots()
        axB.bar(roi_tbl['Channel'], roi_tbl['Attributed Conversions'])
        axB.set_title("Attributed Conversions by Channel")
        plt.setp(axB.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(figB)

elif page == "🧑‍🤝‍🧑 Rep Scorecards (ROI overlay)":
    st.title("Rep Scorecards — ROI Overlay")
    t = apply_gne_filter(touches.copy(), therapy_col='therapy_area')

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_rep = st.selectbox("Select Rep", sorted(hcps['assigned_rep'].unique().tolist()))
    with col2:
        model = st.selectbox("Attribution Model", ["Last Touch","First Touch","Linear","Time Decay"])
    with col3:
        revenue_per_conv = st.slider("Revenue per conversion (mock)", 100, 2000, 500, step=50)

    lookback_days = st.slider("Lookback (days)", 7, 45, 21)

    rep_hcps = hcps[hcps['assigned_rep']==sel_rep]['hcp_id'].tolist()
    st.write(f"Assigned HCPs: {', '.join(rep_hcps)}")

    df = t[t['hcp_id'].isin(rep_hcps)].copy()
    if df.empty:
        st.info("No touches for this rep and filters.")
    else:
        st.subheader("Channel ROI within Rep's Book")
        roi_tbl = compute_roi_table(df, revenue_per_conv, model=model, lookback_days=lookback_days)
        st.dataframe(roi_tbl)

        figA, axA = plt.subplots()
        axA.bar(roi_tbl['Channel'], roi_tbl['ROI'])
        axA.set_title(f"ROI by Channel — {sel_rep}")
        plt.setp(axA.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(figA)

        st.subheader("Field Rep ROI — Attributed to Rep")
        df = df.sort_values(['hcp_id','date'])
        spend_by_rep = df[df['channel']=='Field Rep'].groupby('rep')['cost'].sum().to_dict()
        credit_by_rep = {}
        for hid in df['hcp_id'].unique():
            sub = df[df['hcp_id']==hid]
            for i, row in sub.iterrows():
                if row['conversion'] == 1:
                    conv_date = row['date']
                    prior = sub[(sub['date']<=conv_date) & (sub['date']>=conv_date - pd.Timedelta(days=lookback_days))]
                    prior = prior[prior['conversion']==0]
                    if prior.empty: continue
                    if model == "Last Touch":
                        last = prior.sort_values('date').iloc[-1]
                        if last['channel'] == 'Field Rep':
                            rep_name = last.get('rep','Unknown')
                            credit_by_rep[rep_name] = credit_by_rep.get(rep_name, 0.0) + 1.0
                    elif model == "First Touch":
                        first = prior.sort_values('date').iloc[0]
                        if first['channel'] == 'Field Rep':
                            rep_name = first.get('rep','Unknown')
                            credit_by_rep[rep_name] = credit_by_rep.get(rep_name, 0.0) + 1.0
                    elif model == "Linear":
                        w = 1.0 / len(prior)
                        for j, prow in prior.iterrows():
                            if prow['channel']=='Field Rep':
                                rep_name = prow.get('rep','Unknown')
                                credit_by_rep[rep_name] = credit_by_rep.get(rep_name, 0.0) + w
                    else:
                        weights = []
                        for j, prow in prior.iterrows():
                            age = (conv_date - prow['date']).days
                            weights.append(np.exp(-0.1 * age))
                        total_w = sum(weights) if sum(weights)>0 else 1.0
                        for (j, prow), w in zip(prior.iterrows(), weights):
                            if prow['channel']=='Field Rep':
                                rep_name = prow.get('rep','Unknown')
                                credit_by_rep[rep_name] = credit_by_rep.get(rep_name, 0.0) + (w/total_w)

        reps = sorted(set(list(spend_by_rep.keys()) + list(credit_by_rep.keys())))
        rep_tbl = pd.DataFrame({
            'Rep': reps,
            'FR Attributed Conversions': [credit_by_rep.get(r,0.0) for r in reps],
            'FR Spend': [spend_by_rep.get(r,0.0) for r in reps],
        })
        rep_tbl['FR Revenue'] = rep_tbl['FR Attributed Conversions'] * revenue_per_conv
        rep_tbl['FR ROI'] = np.where(rep_tbl['FR Spend']>0, (rep_tbl['FR Revenue'] - rep_tbl['FR Spend']) / rep_tbl['FR Spend'], 0.0)
        st.dataframe(rep_tbl)

        if not rep_tbl.empty:
            figB, axB = plt.subplots()
            axB.bar(rep_tbl['Rep'], rep_tbl['FR ROI'])
            axB.set_title("Field Rep ROI by Rep (Attributed)")
            plt.setp(axB.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(figB)

elif page == "📚 Content Factory":
    st.title("Content Factory")
    assets_f = apply_gne_filter(assets.copy(), therapy_col='therapy_area')
    ta = ["All"] + sorted(assets_f['therapy_area'].unique().tolist())
    ch = ["All"] + sorted(assets_f['channel'].unique().tolist())
    c1, c2 = st.columns(2)
    with c1: sel_ta = st.selectbox("Therapy Area", ta)
    with c2: sel_ch = st.selectbox("Channel", ch)
    df = assets_f.copy()
    if sel_ta != "All": df = df[df['therapy_area']==sel_ta]
    if sel_ch != "All": df = df[df['channel']==sel_ch]
    st.dataframe(df)

elif page == "🧭 Next Best Action (rules + scoring)":
    st.title("Next Best Action (rules + scoring)")
    sel_hcp = st.selectbox("Select HCP", hcps['name'].tolist())
    hcp_id = hcps[hcps['name']==sel_hcp]['hcp_id'].iloc[0]
    ref_time = interactions['timestamp'].max()
    h_inter = interactions[interactions['hcp_id']==hcp_id].copy()

    def recent_count(df, ch, days=14):
        start = ref_time - pd.Timedelta(days=days)
        return int(df[(df['timestamp']>=start) & (df['channel']==ch)].shape[0])

    prefs = hcps[hcps['hcp_id']==hcp_id]['preferred_channels'].iloc[0].split('|')
    seg = hcps[hcps['hcp_id']==hcp_id]['segments'].iloc[0].lower()
    presc = hcps[hcps['hcp_id']==hcp_id]['prescribing_volume'].iloc[0]

    candidate_channels = ["Email","Portal","Webinar","Field Rep","KOL"]
    def score(ch):
        pref_bonus = 1.0 if ch in prefs else 0.0
        seg_match = 1.0 if (('digital' in seg and ch in ['Email','Webinar','Portal']) or ('in-person' in seg and ch in ['Field Rep','KOL'])) else 0.0
        portal_intensity = recent_count(h_inter, 'Portal', 14)
        portal_boost = portal_intensity * (1.0 if ch in ['Email','Webinar'] else 0.2)
        recency_days = days_since_last(h_inter, ch, ref_time)
        presc_bonus = 0.5 if presc=='High' and ch in ['Field Rep','KOL'] else 0.0
        base = 0.5
        return base + 0.8*pref_bonus + 0.6*seg_match + 0.4*portal_boost + presc_bonus - 0.3*(recency_days/30.0)

    scores = sorted([(ch, score(ch)) for ch in candidate_channels], key=lambda x: x[1], reverse=True)
    topN = st.slider("How many actions to show", 1, 5, 3)
    st.subheader("Top Recommendations")
    for ch, sc in scores[:topN]:
        st.write(f"**{ch}** — score {sc:.2f}")

    st.markdown("**Rationale**")
    st.write(f"- Preferences: {', '.join(prefs)}")
    st.write(f"- Segment: {seg} | Prescribing: {presc}")
    st.write(f"- Portal intensity 14d: {recent_count(h_inter,'Portal',14)}")
    st.write("- Recency (days since last): " + ", ".join([f"{c}:{days_since_last(h_inter,c,ref_time)}" for c in candidate_channels]))

    st.markdown("---")
    st.subheader("Suggested Approved Content")
    assets_f = apply_gne_filter(assets.copy(), therapy_col='therapy_area')
    st.dataframe(assets_f[assets_f['mlr_status']=='Approved'][['asset_id','title','therapy_area','format','channel','tags']].head(6))

elif page == "🧪 Brand/Therapy Filters + Cohorts":
    st.title("Brand/Therapy Filters + Cohorts")
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_spec = st.multiselect("Specialty", sorted(hcps['specialty'].unique().tolist()))
    with c2:
        sel_geo = st.multiselect("Location", sorted(hcps['location'].unique().tolist()))
    with c3:
        sel_presc = st.multiselect("Prescribing Volume", sorted(hcps['prescribing_volume'].unique().tolist()))

    c4, c5 = st.columns(2)
    with c4:
        seg_kw = st.text_input("Segment keyword (digital, in-person, early, etc.)", "")
    with c5:
        t = apply_gne_filter(touches.copy(), therapy_col='therapy_area')
        sel_brand = st.multiselect("Brand", sorted(t['brand'].unique().tolist()))
        sel_therapy = st.multiselect("Therapy Area", sorted(t['therapy_area'].unique().tolist()))

    df = hcps.copy()
    if sel_spec: df = df[df['specialty'].isin(sel_spec)]
    if sel_geo: df = df[df['location'].isin(sel_geo)]
    if sel_presc: df = df[df['prescribing_volume'].isin(sel_presc)]
    if seg_kw: df = df[df['segments'].str.lower().str.contains(seg_kw.lower())]

    tt = t.copy()
    if sel_brand: tt = tt[tt['brand'].isin(sel_brand)]
    if sel_therapy: tt = tt[tt['therapy_area'].isin(sel_therapy)]
    if sel_brand or sel_therapy:
        engaged_hcps = tt['hcp_id'].unique().tolist()
        df = df[df['hcp_id'].isin(engaged_hcps)]

    st.subheader("Cohort Members")
    st.dataframe(df[['hcp_id','name','specialty','location','segments','prescribing_volume','preferred_channels','assigned_rep','last_engagement_date']])

    st.markdown("---")
    st.subheader("Cohort Interaction Mix")
    cohort_inter = interactions[interactions['hcp_id'].isin(df['hcp_id'].tolist())]
    if cohort_inter.empty:
        st.info("No interactions for current cohort filters.")
    else:
        ch_counts = cohort_inter['channel'].value_counts().reset_index()
        ch_counts.columns = ['channel','count']
        st.dataframe(ch_counts)
        fig, ax = plt.subplots()
        ax.bar(ch_counts['channel'], ch_counts['count'])
        ax.set_title("Cohort Interactions by Channel")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

elif page == "💹 Attribution & ROI (global)":
    st.title("Attribution & ROI — Global")
    t = apply_gne_filter(touches.copy(), therapy_col='therapy_area')

    brands = sorted(t['brand'].unique().tolist())
    sel_brands = st.multiselect("Brands (overlay)", brands, default=brands[:2] if brands else [])
    model = st.selectbox("Attribution Model", ["Last Touch","First Touch","Linear","Time Decay"])
    revenue_per_conv = st.slider("Revenue per conversion (mock)", 100, 2000, 500, step=50)
    lookback_days = st.slider("Lookback (days)", 7, 45, 21)

    if not sel_brands:
        st.info("Select at least one brand to show overlays.")
    else:
        for b in sel_brands:
            st.subheader(f"Brand: {b}")
            dfb = t[t['brand']==b]
            if dfb.empty:
                st.info("No data for this brand."); continue
            roi_tbl = compute_roi_table(dfb, revenue_per_conv, model=model, lookback_days=lookback_days)
            st.dataframe(roi_tbl)
            fig1, ax1 = plt.subplots()
            ax1.bar(roi_tbl['Channel'], roi_tbl['ROI'])
            ax1.set_title(f"ROI by Channel — {b} (Model: {model})")
            plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig1)
            fig2, ax2 = plt.subplots()
            ax2.bar(roi_tbl['Channel'], roi_tbl['Attributed Conversions'])
            ax2.set_title(f"Attributed Conversions — {b}")
            plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig2)
