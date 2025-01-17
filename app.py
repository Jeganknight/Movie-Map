import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from imdb import IMDb

# Initialize IMDb
ia = IMDb()

# Streamlit UI
st.title("🎥 Similar Movie Recommendation Assistant")
st.subheader("Step 1: Enter API Key")

# User input for API key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your API key to proceed.")
    st.stop()

# Initialize Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Initialize the prompt templates
synopsis_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable movie recommendation assistant specializing in global cinema.
    Based on the user's provided movie's synopsis and genres, find 6-7 similar movies from the entire IMDb database,
    including their titles, years, countries of origin, storylines, IMDb ratings, and explain why each of the movies is similar to the provided movie.

    Movie Synopsis:
    {storyline}

    Movie IMDb Rating:
    {rating}

    Movie Genres:
    {genres}

    Additional Preferences:
    {content_restrictions}

    Your Response:
    Provide a list of 6-7 movies with their titles, years, countries, IMDb ratings, storylines, and explain the similarity of each movie with the provided movie's storyline.
    """
)

plot_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable movie recommendation assistant specializing in global cinema.
    Based on the user's provided movie's plot and genres, find 6-7 similar movies from the entire IMDb database,
    including their titles, years, countries of origin, storylines, IMDb ratings, and explain why each of the movies is similar to the provided movie.

    Movie Plot:
    {storyline}

    Movie IMDb Rating:
    {rating}

    Movie Genres:
    {genres}

    Additional Preferences:
    {content_restrictions}

    Your Response:
    Provide a list of 6-7 movies with their titles, years, countries, IMDb ratings, storylines, and explain the similarity of each movie with the provided movie's storyline.
    """
)

# Movie name and year input
movie_name = st.text_input("Enter Movie Name")
movie_year = st.number_input("Enter Movie Year", min_value=1900, max_value=2024, step=1)

# Collapsible User Preference Section
with st.expander("Click to Add Your Movie Preferences (Optional)"):
    st.markdown("""
    **Please describe the following in your preferences (Optional):**
    - **Liked Parts**: What you liked about the movie (e.g., suspense, plot twist, atmosphere).
    - **Disliked Parts**: What you didn’t like about the movie (e.g., too violent, too slow).
    - **Content Restrictions**: Any content restrictions (e.g., no violence, no explicit scenes).
    - **Year Range**: Specific year range (e.g., movies from 2000-2010).
    
    **Example Format for Preferences**:
    ```text
    Liked Parts: I enjoy supernatural thrillers with a strong female lead and suspenseful plots. I liked the mystery around the house and the eerie atmosphere.
    Disliked Parts: I don't like overly violent movies. I prefer psychological thrillers.
    Content Restrictions: I prefer PG-13 movies without extreme violence.
    Year Range: Movies from 2000-2010.
    ```
    """)

preferences_input = st.text_area(
    "Enter your movie preferences (liked part, disliked part, content restrictions, year range, etc.)",
    help="Describe the parts of the movie you liked, disliked, content restrictions, and any specific year range you're interested in."
)

if st.button("Get Similar Movie Recommendations"):
    if not movie_name or not movie_year:
        st.error("Please provide both movie name and year.")
        st.stop()

    # Search IMDb for the movie by name and year
    search_results = ia.search_movie(movie_name)
    matching_movie = None

    # Find the movie with the exact match for year
    for movie in search_results:
        if movie.get('year') == movie_year:
            matching_movie = ia.get_movie(movie.movieID)
            break

    if not matching_movie:
        st.error("No matching movie found for the given name and year.")
        st.stop()

    # Fetch the movie's synopsis or plot and IMDb rating
    movie_synopsis = matching_movie.get('synopsis')
    movie_plot = matching_movie.get('plot')
    movie_rating = matching_movie.get('rating')
    movie_genres = ", ".join(matching_movie.get('genres', []))

    # Prepare the user preferences and additional context
    user_input = {
        "rating": movie_rating or "Not available",
        "genres": movie_genres,
        "content_restrictions": "The movie should not contain adult content or explicit scenes. The runtime should be less than 2 hours."  # Default if not provided by the user
    }

    # Process the preferences input
    if preferences_input:
        # Extract relevant information from the preferences input
        liked_part = None
        disliked_part = None
        year_range = None
        content_restrictions = "The movie should not contain adult content or explicit scenes."  # Default

        # If the user described liked/disliked parts
        if "liked" in preferences_input.lower():
            liked_part = preferences_input.lower()

        if "disliked" in preferences_input.lower():
            disliked_part = preferences_input.lower()

        # Extract year range if provided
        if "year range" in preferences_input.lower():
            year_range = preferences_input.lower()

        if "content restrictions" in preferences_input.lower():
            content_restrictions = preferences_input.lower()

        # Based on the input, create the appropriate prompt
        user_input["content_restrictions"] = content_restrictions
        
        if liked_part:
            user_input["storyline"] = liked_part
            prompt_template = plot_prompt_template if movie_plot else synopsis_prompt_template
            prompt = prompt_template.format(**user_input)
        else:
            if movie_synopsis:
                user_input["storyline"] = movie_synopsis
                prompt = synopsis_prompt_template.format(**user_input)
            elif movie_plot:
                user_input["storyline"] = movie_plot[0]
                prompt = plot_prompt_template.format(**user_input)

        try:
            response = llm.invoke(prompt)
            st.subheader("Similar Movie Recommendations Based on Your Preferences:")
            st.write(response.content.strip())
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        if movie_synopsis:
            user_input["storyline"] = movie_synopsis
            prompt = synopsis_prompt_template.format(**user_input)
        elif movie_plot:
            user_input["storyline"] = movie_plot[0]
            prompt = plot_prompt_template.format(**user_input)

        try:
            response = llm.invoke(prompt)
            st.subheader("Similar Movie Recommendations Based on the Movie's Plot/Synopsis:")
            st.write(response.content.strip())
        except Exception as e:
            st.error(f"An error occurred: {e}")
