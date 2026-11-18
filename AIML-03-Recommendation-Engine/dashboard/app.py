
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation Engine")
st.write(
    "Discover popular movies and get recommendations "
    "that exclude movies you've already watched."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Recommendation Settings")

user_id = st.sidebar.number_input(
    "MovieLens User ID",
    min_value=1,
    value=1,
    step=1
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

recommendation_type = st.sidebar.radio(
    "Choose recommendation type",
    ["For a User", "Popular Movies"]
)

if st.button("Get Recommendations", type="primary"):

    try:
        with st.spinner("Finding movies for you..."):

            if recommendation_type == "For a User":
                response = requests.get(
                    f"{API_URL}/recommend/{user_id}",
                    params={"top_k": top_k},
                    timeout=30
                )
            else:
                response = requests.get(
                    f"{API_URL}/popular",
                    params={"top_k": top_k},
                    timeout=30
                )

            response.raise_for_status()
            data = response.json()

        recommendations = data["recommendations"]

        if not recommendations:
            st.warning("No recommendations available.")
        else:
            st.success(
                f"Found {len(recommendations)} recommendations!"
            )

            if recommendation_type == "For a User":
                st.caption(
                    f"Recommendations for MovieLens user {user_id}"
                )
            else:
                st.caption("General popular movie recommendations")

            for movie in recommendations:

                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.subheader(
                            f"{movie['rank']}. {movie['title']}"
                        )
                        st.write(f"**Genres:** {movie['genres']}")

                    with col2:
                        st.metric(
                            "Average rating",
                            f"{movie['average_rating']:.2f}/5"
                        )
                        st.caption(
                            f"{movie['rating_count']:,} ratings"
                        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the recommendation API. "
            "Please start the FastAPI server first."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")

    except Exception as error:
        st.error(f"Something went wrong: {error}")

st.divider()

st.caption(
    "MovieLens 25M Dataset | Popularity-Based Recommendation "
    "with Watched-Movie Filtering"
)
