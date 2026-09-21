import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tamil Nadu Scheme Assistant",
    page_icon="🇮🇳",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title(
    "🇮🇳 Tamil Nadu Government Scheme Assistant"
)

st.write(
    "Search Tamil Nadu Government schemes using "
    "AI-powered hybrid retrieval."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "💬 Conversation"
    )

    st.write(
        "Ask follow-up questions and the assistant "
        "will use the previous conversation as context."
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content"
    )

    if role == "user":

        with st.chat_message("user"):

            st.write(content)

    elif role == "assistant":

        with st.chat_message("assistant"):

            st.markdown(content)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask about Tamil Nadu Government schemes..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)

    # --------------------------------------------------------
    # Prepare conversation history
    # --------------------------------------------------------

    conversation_history = (
        st.session_state.messages.copy()
    )

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------------------------
    # Call backend
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching Government schemes..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",

                    json={
                        "question": question,

                        "conversation_history": (
                            conversation_history
                        )
                    },

                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend."
                )

                st.info(
                    "Make sure this is running:\n\n"
                    "`uvicorn backend.app.main:app --reload`"
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long."
                )

                st.stop()

            except requests.exceptions.RequestException as e:

                st.error(
                    f"API request failed: {e}"
                )

                st.stop()

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            "No answer was returned."
        )

        st.markdown(answer)

        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        filters = result.get(
            "filters",
            {}
        )

        active_filters = {
            key: value
            for key, value in filters.items()
            if value
        }

        if active_filters:

            st.divider()

            st.subheader(
                "🔎 Detected Filters"
            )

            columns = st.columns(
                min(
                    len(active_filters),
                    3
                )
            )

            for index, (
                key,
                value
            ) in enumerate(
                active_filters.items()
            ):

                with columns[
                    index % len(columns)
                ]:

                    display_name = (
                        key
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    st.metric(
                        display_name,
                        value
                    )

        # ----------------------------------------------------
        # RETRIEVAL INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Retrieval Information"
        )

        columns = st.columns(4)

        with columns[0]:

            st.metric(
                "Vector",
                result.get(
                    "vector_results",
                    0
                )
            )

        with columns[1]:

            st.metric(
                "Graph",
                result.get(
                    "graph_results",
                    0
                )
            )

        with columns[2]:

            st.metric(
                "Structured",
                result.get(
                    "structured_results",
                    0
                )
            )

        with columns[3]:

            st.metric(
                "Combined",
                result.get(
                    "combined_results",
                    0
                )
            )


        # ----------------------------------------------------
        # SCHEMES
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        st.divider()

        st.subheader(
            f"📋 Matching Schemes ({len(sources)})"
        )

        if not sources:

            st.info(
                "No matching schemes were found."
            )

        else:

            # ------------------------------------------------
            # Two-column scheme cards
            # ------------------------------------------------

            for start in range(
                0,
                len(sources),
                2
            ):

                columns = st.columns(2)

                row_sources = sources[
                    start:start + 2
                ]

                for column, source in zip(
                    columns,
                    row_sources
                ):

                    with column:

                        # ------------------------------------
                        # TITLE
                        # ------------------------------------

                        title = source.get(
                            "title",
                            "Unknown Scheme"
                        )

                        st.markdown(
                            f"### 📜 {title}"
                        )

                        # ------------------------------------
                        # BASIC INFORMATION
                        # ------------------------------------

                        department = source.get(
                            "department"
                        )

                        beneficiary = source.get(
                            "beneficiary"
                        )

                        benefit_type = source.get(
                            "benefit_type"
                        )

                        sponsored_by = source.get(
                            "sponsored_by"
                        )

                        if department:

                            st.write(
                                f"🏢 **Department:** "
                                f"{department}"
                            )

                        if beneficiary:

                            st.write(
                                f"👥 **Beneficiary:** "
                                f"{beneficiary}"
                            )

                        if benefit_type:

                            st.write(
                                f"💰 **Benefit Type:** "
                                f"{benefit_type}"
                            )

                        if sponsored_by:

                            st.write(
                                f"🏛️ **Sponsored By:** "
                                f"{sponsored_by}"
                            )

                        # ------------------------------------
                        # DESCRIPTION
                        # ------------------------------------

                        description = source.get(
                            "description"
                        )

                        if description:

                            st.write(
                                "**Description**"
                            )

                            st.write(
                                description
                            )

                        # ------------------------------------
                        # FULL DETAILS
                        # ------------------------------------

                        with st.expander(
                            "📖 View Full Scheme Details"
                        ):

                            eligibility = source.get(
                                "eligibility"
                            )

                            how_to_avail = source.get(
                                "how_to_avail"
                            )

                            district = source.get(
                                "district"
                            )

                            organisation = source.get(
                                "organisation"
                            )

                            associated_scheme = (
                                source.get(
                                    "associated_scheme"
                                )
                            )

                            funding_pattern = (
                                source.get(
                                    "funding_pattern"
                                )
                            )

                            validity = source.get(
                                "validity"
                            )

                            introduced_on = (
                                source.get(
                                    "introduced_on"
                                )
                            )

                            scheme_type = (
                                source.get(
                                    "scheme_type"
                                )
                            )

                            uploaded_file = (
                                source.get(
                                    "uploaded_file"
                                )
                            )

                            # --------------------------------
                            # Eligibility
                            # --------------------------------

                            if eligibility:

                                st.markdown(
                                    "#### ✅ Eligibility"
                                )

                                st.write(
                                    eligibility
                                )

                            # --------------------------------
                            # How to avail
                            # --------------------------------

                            if how_to_avail:

                                st.markdown(
                                    "#### 📋 How to Avail"
                                )

                                st.write(
                                    how_to_avail
                                )

                            # --------------------------------
                            # District
                            # --------------------------------

                            if district:

                                st.markdown(
                                    "#### 📍 District"
                                )

                                st.write(
                                    district
                                )

                            # --------------------------------
                            # Organisation
                            # --------------------------------

                            if organisation:

                                st.markdown(
                                    "#### 🏢 Organisation"
                                )

                                st.write(
                                    organisation
                                )

                            # --------------------------------
                            # Associated scheme
                            # --------------------------------

                            if associated_scheme:

                                st.markdown(
                                    "#### 🔗 Associated Scheme"
                                )

                                st.write(
                                    associated_scheme
                                )

                            # --------------------------------
                            # Funding
                            # --------------------------------

                            if funding_pattern:

                                st.markdown(
                                    "#### 💰 Funding Pattern"
                                )

                                st.write(
                                    funding_pattern
                                )

                            # --------------------------------
                            # Validity
                            # --------------------------------

                            if validity:

                                st.markdown(
                                    "#### 📅 Validity"
                                )

                                st.write(
                                    validity
                                )

                            # --------------------------------
                            # Introduced
                            # --------------------------------

                            if introduced_on:

                                st.markdown(
                                    "#### 🗓️ Introduced On"
                                )

                                st.write(
                                    introduced_on
                                )

                            # --------------------------------
                            # Scheme Type
                            # --------------------------------

                            if scheme_type:

                                st.markdown(
                                    "#### 🏷️ Scheme Type"
                                )

                                st.write(
                                    scheme_type
                                )

                            # --------------------------------
                            # Uploaded File
                            # --------------------------------

                            if uploaded_file:

                                st.markdown(
                                    "#### 📎 Document"
                                )

                                st.write(
                                    uploaded_file
                                )

                        # ------------------------------------
                        # RETRIEVAL SOURCE
                        # ------------------------------------

                        retrieved_by = source.get(
                            "retrieved_by"
                        )

                        if retrieved_by:

                            st.caption(
                                f"Retrieved by: "
                                f"{retrieved_by}"
                            )

                        # ------------------------------------
                        # GOVERNMENT SOURCE
                        # ------------------------------------

                        url = source.get(
                            "url"
                        )

                        if url:

                            st.link_button(
                                "🔗 View Official Government Source",
                                url,
                                use_container_width=True
                            )

                        st.divider()