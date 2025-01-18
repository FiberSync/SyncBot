import streamlit as st

# Create a sidebar with multiple pages
st.sidebar.title("Navigation")

pages = ["Home", "About", "Contact", "FAQ"]

selected_page = st.sidebar.selectbox("Select a page", pages)

# Define the content for each page
if selected_page == "Home":
    st.title("Welcome to our app!")
    st.write("This is the home page.")
elif selected_page == "About":
    st.title("About us")
    st.write("This is the about page.")
elif selected_page == "Contact":
    st.title("Contact us")
    st.write("This is the contact page.")
elif selected_page == "FAQ":
    st.title("Frequently Asked Questions")
    st.write("This is the FAQ page.")

# You can also use st.sidebar.button to create buttons for each page
# st.sidebar.button("Home")
# st.sidebar.button("About")
# st.sidebar.button("Contact")
# st.sidebar.button("FAQ")