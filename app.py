from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
import streamlit as st
import time
from dotenv import load_dotenv

load_dotenv()

hugging_face_api_key = os.getenv("HF_TOKEN")

if not hugging_face_api_key:
    raise ValueError("❌ Missing HF_TOKEN. Please set it in your .env or repo secrets.")

MODEL_OPTIONS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3.1-8B-Instruct"
]

@st.cache_resource
def initialize_langchain_model(model_id):
    """Initialize LangChain model with proper error handling."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            return_full_text=False,
            huggingfacehub_api_token=hugging_face_api_key,
        )
        
        chat_model = ChatHuggingFace(llm=llm)
        return chat_model, "success"
    
    except Exception as e:
        return None, str(e)

def safe_invoke_chain(chain, inputs, max_retries=3):
    """Safely invoke a LangChain chain with retries and error handling."""
    for attempt in range(max_retries):
        try:
            response = chain.invoke(inputs)
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict) and 'text' in response:
                return response['text'].strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        
        except StopIteration:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ StopIteration error (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2)
                continue
            else:
                return "Error: Model failed after multiple attempts. Please try a different model."
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    st.warning(f"⚠️ Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Rate limit exceeded. Please wait a minute and try again."
            
            elif "timeout" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Timeout (attempt {attempt + 1}/{max_retries}). Retrying...")
                    time.sleep(3)
                    continue
                else:
                    return "Error: Request timed out. Try simplifying your request."
            
            elif "503" in error_msg or "loading" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Model loading (attempt {attempt + 1}/{max_retries}). Please wait...")
                    time.sleep(5)
                    continue
                else:
                    return "Error: Model is currently loading. Please try again in a minute."
            
            else:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Error: {str(e)}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    return f"Error: {str(e)}"
    
    return "Error: All retry attempts failed."

def create_recipe_generation_chain(chat_model):
    """Create a LangChain chain for recipe generation."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert chef and recipe creator. Create delicious, practical, and easy-to-follow recipes. Be creative but realistic with ingredient combinations."),
        ("human", """Create a detailed recipe with the following specifications:

Ingredients available: {ingredients}
Cuisine type: {cuisine}
Dietary restrictions: {dietary}
Cooking time preference: {cooking_time}
Difficulty level: {difficulty}

Please provide:
1. Recipe name (creative and appetizing)
2. Servings
3. Prep time and cook time
4. Complete ingredient list with measurements
5. Step-by-step cooking instructions (numbered)
6. Cooking tips and variations
7. Nutritional information (if applicable)

Make it detailed, clear, and beginner-friendly.""")
    ])
    
    chain = prompt | chat_model
    return chain

def create_recipe_ideas_chain(chat_model):
    """Create a chain for generating multiple recipe ideas."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative chef who suggests recipe ideas based on available ingredients."),
        ("human", """Based on these ingredients: {ingredients}

Generate 5 creative recipe ideas (just names and brief descriptions, 2-3 sentences each).
Consider cuisine type: {cuisine}
Dietary restrictions: {dietary}

Format:
1. [Recipe Name] - [Brief description]
2. [Recipe Name] - [Brief description]
etc.""")
    ])
    
    chain = prompt | chat_model
    return chain

def generate_recipe_ideas(ingredients, cuisine, dietary, chat_model):
    """Generate recipe ideas."""
    chain = create_recipe_ideas_chain(chat_model)
    return safe_invoke_chain(chain, {
        "ingredients": ingredients,
        "cuisine": cuisine,
        "dietary": dietary
    })

def generate_recipe(ingredients, cuisine, dietary, cooking_time, difficulty, chat_model):
    """Generate full recipe."""
    chain = create_recipe_generation_chain(chat_model)
    return safe_invoke_chain(chain, {
        "ingredients": ingredients,
        "cuisine": cuisine,
        "dietary": dietary,
        "cooking_time": cooking_time,
        "difficulty": difficulty
    })

# Streamlit UI
st.set_page_config(page_title="AI Recipe Generator", page_icon="👨‍🍳", layout="wide")

st.title("👨‍🍳 AI Recipe Generator")
st.header("Transform Your Ingredients Into Delicious Meals")

# Initialize session state
if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None
    st.session_state['current_model_name'] = MODEL_OPTIONS[0]
    st.session_state['model_status'] = "not_loaded"

if 'available_ingredients' not in st.session_state:
    st.session_state['available_ingredients'] = []

# Sidebar for settings
with st.sidebar:
    st.header("🤖 Model Settings")
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state['current_model_name']) if st.session_state['current_model_name'] in MODEL_OPTIONS else 0,
        help="Zephyr and Mistral work best for recipes"
    )
    
    if st.button("🔄 Load Model", type="primary", use_container_width=True):
        with st.spinner(f"Loading {selected_model}..."):
            model, status = initialize_langchain_model(selected_model)
            if status == "success":
                st.session_state['current_model'] = model
                st.session_state['current_model_name'] = selected_model
                st.session_state['model_status'] = "loaded"
                st.success(f"✅ Model loaded successfully!")
                st.rerun()
            else:
                st.session_state['model_status'] = "error"
                st.error(f"❌ Failed to load model: {status}")
    
    st.divider()
    
    if st.session_state['model_status'] == "loaded":
        st.success(f"✅ Active: {st.session_state['current_model_name'].split('/')[-1]}")
    elif st.session_state['model_status'] == "error":
        st.error("❌ Model not loaded")
    else:
        st.info("ℹ️ Please load a model to start")
    
    st.divider()
    st.header("🍽️ Quick Tips")
    st.write("""
    - Add at least 3-5 ingredients
    - Be specific (e.g., "boneless chicken" vs "chicken")
    - Include basics like oil, salt, spices
    - Try different cuisines for variety
    """)
    
    if st.button("🗑️ Clear All Cache", use_container_width=True):
        st.cache_resource.clear()
        for key in ['recipe_ideas', 'full_recipe', 'available_ingredients']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared!")
        st.rerun()

if st.session_state['model_status'] != "loaded":
    st.warning("⚠️ Please load a model from the sidebar to begin generating recipes.")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

# Left column - Input ingredients and preferences
with col1:
    st.subheader("🥗 Step 1: Your Kitchen")
    
    with st.container(border=True):
        st.write("**Add Your Ingredients:**")
        
        col_ing1, col_ing2 = st.columns([3, 1])
        with col_ing1:
            ingredient_input = st.text_input(
                "Ingredient:",
                placeholder="e.g., chicken breast, tomatoes, pasta",
                key="ingredient_input",
                label_visibility="collapsed"
            )
        with col_ing2:
            if st.button("➕ Add", use_container_width=True, key="add_ingredient"):
                if ingredient_input.strip() and ingredient_input.strip() not in st.session_state['available_ingredients']:
                    st.session_state['available_ingredients'].append(ingredient_input.strip())
                    st.rerun()
        
        # Display ingredients as tags
        if st.session_state['available_ingredients']:
            st.write("**Your Ingredients:**")
            ingredients_html = " ".join([
                f'<span style="background-color:#FF6B6B;color:white;padding:5px 12px;margin:3px;border-radius:20px;display:inline-block;">🥕 {ing}</span>'
                for ing in st.session_state['available_ingredients']
            ])
            st.markdown(ingredients_html, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear All Ingredients", key="clear_ingredients"):
                st.session_state['available_ingredients'] = []
                st.rerun()
        else:
            st.info("👆 Add ingredients above to get started")
    
    st.write("")
    
    with st.container(border=True):
        st.write("**Preferences & Restrictions:**")
        
        cuisine_type = st.selectbox(
            "Cuisine Type:",
            ["Any", "Italian", "Indian", "Chinese", "Mexican", "Mediterranean", 
             "Thai", "Japanese", "American", "French", "Middle Eastern"],
            index=0
        )
        
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions:",
            ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", 
             "Nut-Free", "Low-Carb", "Keto", "Paleo"],
            default=["None"]
        )
        
        cooking_time = st.select_slider(
            "Cooking Time:",
            options=["Under 15 min", "15-30 min", "30-45 min", "45-60 min", "60+ min"],
            value="30-45 min"
        )
        
        difficulty_level = st.radio(
            "Difficulty Level:",
            ["Easy", "Medium", "Hard"],
            horizontal=True,
            index=0
        )
    
    st.write("")
    
    # Generate ideas button
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        generate_ideas_btn = st.button(
            "💡 Get Recipe Ideas", 
            type="primary", 
            use_container_width=True,
            disabled=len(st.session_state['available_ingredients']) < 2
        )
    with col_btn2:
        if st.button("🗑️ Clear Ideas", use_container_width=True):
            if 'recipe_ideas' in st.session_state:
                del st.session_state['recipe_ideas']
                st.rerun()

# Right column - Recipe ideas and generation
with col2:
    st.subheader("📋 Step 2: Recipe Ideas")
    
    if generate_ideas_btn:
        if len(st.session_state['available_ingredients']) >= 2:
            ingredients_str = ", ".join(st.session_state['available_ingredients'])
            dietary_str = ", ".join(dietary_restrictions)
            
            with st.spinner('🔄 Generating recipe ideas... (30-60 seconds)'):
                ideas = generate_recipe_ideas(
                    ingredients_str,
                    cuisine_type,
                    dietary_str,
                    st.session_state['current_model']
                )
                st.session_state['recipe_ideas'] = ideas
                st.rerun()
        else:
            st.warning("⚠️ Please add at least 2 ingredients first.")
    
    # Display recipe ideas
    if 'recipe_ideas' in st.session_state:
        with st.container(border=True):
            if st.session_state['recipe_ideas'].startswith("Error"):
                st.error(st.session_state['recipe_ideas'])
                st.info("💡 Try: Loading a different model or reducing ingredients")
            else:
                st.success("✅ Recipe ideas generated!")
                st.text_area(
                    "Recipe Ideas:",
                    value=st.session_state['recipe_ideas'],
                    height=250,
                    key="ideas_display"
                )
    
    st.write("")
    st.subheader("👨‍🍳 Step 3: Full Recipe")
    
    with st.container(border=True):
        selected_recipe = st.text_input(
            "Selected Recipe Name:",
            placeholder="Type or paste recipe name from ideas above",
            key="selected_recipe",
            help="You can also create your own recipe name"
        )
        
        col_gen1, col_gen2 = st.columns([1, 1])
        with col_gen1:
            generate_recipe_btn = st.button(
                "🚀 Generate Full Recipe",
                type="primary",
                use_container_width=True,
                disabled=len(st.session_state['available_ingredients']) < 2
            )
        with col_gen2:
            if st.button("🗑️ Clear Recipe", use_container_width=True):
                if 'full_recipe' in st.session_state:
                    del st.session_state['full_recipe']
                    st.rerun()

if generate_recipe_btn:
    if len(st.session_state['available_ingredients']) >= 2:
        ingredients_str = ", ".join(st.session_state['available_ingredients'])
        dietary_str = ", ".join(dietary_restrictions)
        
        with st.spinner('🔄 Creating your recipe... (1-2 minutes)'):
            recipe = generate_recipe(
                ingredients_str,
                cuisine_type,
                dietary_str,
                cooking_time,
                difficulty_level,
                st.session_state['current_model']
            )
            st.session_state['full_recipe'] = {
                'name': selected_recipe if selected_recipe else "Custom Recipe",
                'content': recipe,
                'ingredients': ingredients_str,
                'cuisine': cuisine_type,
                'dietary': dietary_str
            }
            st.rerun()
    else:
        st.warning("⚠️ Please add at least 2 ingredients first.")

# Display full recipe (full width at bottom)
if 'full_recipe' in st.session_state:
    st.divider()
    st.subheader("📖 Your Recipe")
    
    if st.session_state['full_recipe']['content'].startswith("Error"):
        st.error(st.session_state['full_recipe']['content'])
        st.info("💡 Try: Reducing ingredients, loading different model, or simplifying preferences")
    else:
        with st.container(border=True):
            st.markdown(f"### {st.session_state['full_recipe']['name']}")
            st.caption(f"🌍 Cuisine: {st.session_state['full_recipe']['cuisine']} | 🥗 Dietary: {st.session_state['full_recipe']['dietary']}")
            st.markdown("---")
            st.write(st.session_state['full_recipe']['content'])
        
        # Download options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            recipe_txt = f"# {st.session_state['full_recipe']['name']}\n\nCuisine: {st.session_state['full_recipe']['cuisine']}\nDietary: {st.session_state['full_recipe']['dietary']}\n\n{st.session_state['full_recipe']['content']}"
            st.download_button(
                label="📥 Download TXT",
                data=recipe_txt,
                file_name=f"{st.session_state['full_recipe']['name'].replace(' ', '_')[:30]}_recipe.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            recipe_md = f"# {st.session_state['full_recipe']['name']}\n\n**Cuisine:** {st.session_state['full_recipe']['cuisine']}  \n**Dietary:** {st.session_state['full_recipe']['dietary']}\n\n---\n\n{st.session_state['full_recipe']['content']}"
            st.download_button(
                label="📥 Download MD",
                data=recipe_md,
                file_name=f"{st.session_state['full_recipe']['name'].replace(' ', '_')[:30]}_recipe.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.success("✅ Recipe generated successfully! Enjoy cooking! 👨‍🍳")

# Footer
st.divider()
st.caption("🦜🔗 Powered by LangChain + Hugging Face | Built with Streamlit | Happy Cooking! 🍳")
