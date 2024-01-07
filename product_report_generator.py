# Import Statements
from dotenv import load_dotenv
import os
import requests
import re
from datetime import datetime
from transformers import pipeline
from textblob import TextBlob
from openai import OpenAI
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.units import inch

# Get credentials from .env file
load_dotenv()
rapid_api_key = os.getenv('RapidAPI_KEY')
opanai_api_key = os.getenv('OpenAi_KEY')


# Extracts the ASIN from an Amazon url
def extract_asin(url):
    match = re.search(r'/dp/(\w+)', url)
    return match.group(1) if match else None


# Gets amazon reviews using rapid api amazon api. 
def get_amazon_reviews(asin, country='US', max_pages=10):
    url = 'https://real-time-amazon-data.p.rapidapi.com/product-reviews'
    headers = {
        'X-RapidAPI-Key': rapid_api_key,
        'X-RapidAPI-Host': 'real-time-amazon-data.p.rapidapi.com'
    }

    all_reviews = []
    for page in range(1, max_pages + 1):
        params = {
            'asin': asin,
            'country': country,
            'page': str(page)
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            page_reviews = response.json().get('data', {}).get('reviews', [])
            all_reviews.extend(page_reviews)
        else:
            print("Error on page", page, ":", response.status_code, response.text)
            break  # Stop fetching if an error occurs

    return all_reviews


# Gets product details using rapid api amazon api. 
def get_product_details(asin, country='US'):
    url = 'https://real-time-amazon-data.p.rapidapi.com/product-details'

    headers = {
        'X-RapidAPI-Key': rapid_api_key,
        'X-RapidAPI-Host': 'real-time-amazon-data.p.rapidapi.com'
    }

    params = {
        'asin': asin,
        'country': country
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"


# Analyzes sentiment of a string. Utilizes Hugging Face sentiment analysis pipeline.
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']



# Process reviews into easier to work with format. Output is a list of dictionaries, each item being a review. 
def process_reviews(reviews):
    structured_reviews = []

    for review in reviews:
        if review.get('review_comment') is None:
            continue

        # Tokenize the comment and check if it exceeds 500 tokens
        comment = review.get('review_comment', 'No Comment')
        if len(comment.split()) > 500:
            continue  # Skip this review

        helpful_votes_text = review.get('helpful_vote_statement', 'No Votes Info')
        match = re.search(r'(\d+)', helpful_votes_text)
        helpful_votes = int(match.group(1)) if match else 0

        date_text = review.get('review_date', 'No Date')
        match = re.search(r'on\s+(\w+\s+\d{1,2},\s+\d{4})', date_text)
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, '%B %d, %Y')
            except ValueError:
                date = None
        else:
            date = None

        rating = int(review.get('review_star_rating', 0))

        structured_review = {
            'title': review.get('review_title', 'No Title'),
            'comment': comment,
            'rating': rating,
            'author': review.get('review_author', 'Anonymous'),
            'date': date,
            'verified_purchase': review.get('is_verified_purchase', False),
            'helpful_votes': helpful_votes
        }
        structured_reviews.append(structured_review)

    for review in structured_reviews:
        sentiment, score = analyze_sentiment(review['comment'])
        review['sentiment'] = sentiment
        review['classification_confidence'] = score

    return structured_reviews

# Split reviews into chunks to fit within the token limit.
def split_reviews(reviews, max_length=8000):
    chunks = []
    current_chunk = ""
    for review in reviews:
        if len(current_chunk) + len(review) > max_length:
            chunks.append(current_chunk)
            current_chunk = review
        else:
            current_chunk += " " + review
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_takeaways(chunk):
    bullet_prompt = f"List the main features or aspects of the product mentioned in the following reviews as concise bullet points:\n\n{chunk}"
    response = client.chat.completions.create(
        model="gpt-4",  # Adjust the model as needed
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": bullet_prompt}]
    )
    return response.choices[0].message.content.strip()

# Parse the bullet points from the given text
def parse_bullet_points(text):
    bullet_points = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('-') or line.startswith('*'):
            bullet_points.append(line)
    return bullet_points

# Combine takeaways from all chunks into a list of bullet points.
def combine_takeaways(takeaways):
    bullet_points = []
    for takeaway in takeaways:
        bullet_points.extend(parse_bullet_points(takeaway))
    return bullet_points


# Condense the list of takeaways into a shorter, more concise list.
def condense_takeaways(takeaways):
    bullet_points = "\n".join(takeaways)
    condense_prompt = (
        f"Please condense the following bullet points into a shorter list with the most important information. "
        f"Limit the list to a maximum of 15 bullet points and limit each bullet point to 10 words or less. Mark each bullet with 'PRO' if it's positive, "
        f"and with 'CON' if it's negative :\n\n{bullet_points}"
    )
    response = client.completions.create(
        model="text-davinci-003",
        prompt=condense_prompt,
        max_tokens=600
    )
    # Parse the response text to separate into PRO and CON takeaways
    final_condensed_takeaways = parse_pro_con_bullet_points(response.choices[0].text.strip())
    return final_condensed_takeaways


# Parse and separate the PRO/CON bullet points into a dictionary.
def parse_pro_con_bullet_points(text, max_items=15):
    pro_con_dict = {'pro_takeaways': [], 'con_takeaways': []}

    # Add items to their respective lists, filtering out those longer than 10 words
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('PRO') or line.startswith('CON'):
            items = [item.strip() for item in line[4:].split(',') if item]
            filtered_items = [item for item in items if len(item.split()) <= 10]

            if line.startswith('PRO'):
                pro_con_dict['pro_takeaways'].extend(filtered_items)
            elif line.startswith('CON'):
                pro_con_dict['con_takeaways'].extend(filtered_items)

    # Next, trim the lists to ensure the total count does not exceed max_items
    total_items = len(pro_con_dict['pro_takeaways']) + len(pro_con_dict['con_takeaways'])
    while total_items > max_items:
        if len(pro_con_dict['pro_takeaways']) > len(pro_con_dict['con_takeaways']):
            pro_con_dict['pro_takeaways'].pop()
        else:
            pro_con_dict['con_takeaways'].pop()
        total_items -= 1

    return pro_con_dict




# USAGE
product_url = input("Enter the Amazon product URL: ")
asin = extract_asin(product_url)
if asin:
    reviews = get_amazon_reviews(asin)

# Gets product details
product_details = get_product_details(asin, country='US')


# Gets structured reviews variable
structured_reviews = process_reviews(reviews)

# Calculating Sentiment Approval Rating
positive_reviews_count = sum(1 for review in structured_reviews if review['sentiment'] == 'POSITIVE')
Sentiment_Approval_Rating = (positive_reviews_count / len(structured_reviews)) * 100


# Concatenate all reviews for analysis and gets the Overall Sentiment Index. 
all_reviews_text = " ".join([review['comment'] for review in structured_reviews])
aggregate_analysis = TextBlob(all_reviews_text)
aggregate_sentiment_score = aggregate_analysis.sentiment.polarity
Overall_Sentiment_Index = (aggregate_sentiment_score + 1) / 2 * 100

client = OpenAI(api_key=opanai_api_key)

all_reviews = [review['comment'] for review in structured_reviews]
review_chunks = split_reviews(all_reviews)
chunk_takeaways = [get_takeaways(chunk) for chunk in review_chunks]
all_takeaways = combine_takeaways(chunk_takeaways)
final_condensed_takeaways = condense_takeaways(all_takeaways)

# Get dataframe of reviews for ease of graphing
df = pd.DataFrame(structured_reviews)

# Code to create graphs for report, they are saved as png files in wrkdir

# Rating Distribution Chart
df_2 = pd.merge(pd.DataFrame({'rating': [1, 2, 3, 4, 5]}), df, on='rating', how='outer').fillna(0)
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=df_2, palette='viridis')
plt.xlabel('Rating (1-5 Stars)')
plt.ylabel('Number of Reviews')
plt.xticks(range(5), ['1', '2', '3', '4', '5'])
plt.savefig('rating_distribution_chart.png')
plt.close()

# Verified Purchases Chart
verified_counts = df['verified_purchase'].value_counts()
plt.figure()
plt.pie(verified_counts, colors=['blue', 'red'], startangle=90)
plt.axis('equal')
legend_labels = [f'{label}: {count} ({percent:.1f}%)' for label, count, percent in 
                 zip(verified_counts.index.map({True: 'Verified', False: 'Non-Verified'}), 
                     verified_counts, 
                     100 * verified_counts / verified_counts.sum())]
plt.legend(legend_labels, title='Purchases', bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.savefig('verified_purchases_chart.png')
plt.close()

# Ratings Over Time Chart
plt.figure(figsize=(12, 6))
sns.scatterplot(x='date', y='rating', data=df)
plt.xlabel('Date')
plt.ylabel('Star Rating')
plt.xticks(rotation=45)
plt.yticks(range(1, 6))
plt.savefig('ratings_over_time_chart.png')
plt.close()


# Code for report

# Creates the initial pdf for the report
def create_pdf(path):
    c = canvas.Canvas(path, pagesize=landscape(letter))
    width, height = landscape(letter)

    # Title
    title = "Amazon Product Analysis Report"
    title_font = "Helvetica-Bold"
    title_font_size = 16
    title_width = c.stringWidth(title, title_font, title_font_size)
    title_x = (width - title_width) / 2
    title_y = height - 50

    c.setFont(title_font, title_font_size)
    c.drawString(title_x, title_y, title)

    return c

# Adds the final takeways to the report
def add_takeaways_section(canvas, takeaways, start_x, start_y, end_x):
    title_font_size = 16
    takeaway_font_size = 10
    title_y_offset = 30
    box_padding = 15
    gap_between_boxes = 10  
    extra_padding = 20  

    # Positive Takeaways title
    canvas.setFont("Helvetica-Bold", title_font_size)
    canvas.setFillColor(colors.green)
    y_pos_title = start_y + title_y_offset
    canvas.drawString(start_x, y_pos_title, "Positive Takeaways:")
    
    # Positive Takeaways content
    canvas.setFont("Helvetica", takeaway_font_size)
    y_pos = y_pos_title - 20
    for pos_takeaway in takeaways['pro_takeaways']:
        canvas.drawString(start_x, y_pos, pos_takeaway)
        y_pos -= 15

    # Negative Takeaways title 
    y_neg_title = y_pos - title_y_offset - gap_between_boxes
    canvas.setFillColor(colors.red)
    canvas.setFont("Helvetica-Bold", title_font_size)
    canvas.drawString(start_x, y_neg_title, "Negative Takeaways:")
    
    # Negative Takeaways content
    y_neg = y_neg_title - 20
    canvas.setFont("Helvetica", takeaway_font_size)
    for neg_takeaway in takeaways['con_takeaways']:
        canvas.drawString(start_x, y_neg, neg_takeaway)
        y_neg -= 15

    top_margin = y_pos_title + extra_padding
    bottom_margin = -y_neg + extra_padding
    total_box_height = top_margin + bottom_margin
    canvas.setStrokeColor(colors.lightblue)
    canvas.rect(start_x - 15, y_pos_title + extra_padding, end_x - start_x + 30, -total_box_height, stroke=1, fill=0)

    # Key Takeaways title 
    title_y_position = y_pos_title + extra_padding + 5  # Adjusting title position above the box
    canvas.setFont("Helvetica-Bold", title_font_size)
    canvas.setFillColor(colors.black)  # Changed color for visibility
    canvas.drawCentredString((start_x + end_x) / 2, title_y_position, "Key Takeaways")


# Adds the Overall Sentiment Index and Sentiment Approval Rating scores to the report
def add_scores_section(canvas, sentiment_index, approval_rating, start_x, start_y, end_x):
    score_name_font_size = 16
    score_value_font_size = 25
    score_desc_font_size = 10
    score_desc_line_height = 10 

    def get_score_color(score):
        # Blue scale for scores
        if score >= 75:
            return colors.darkblue
        elif score >= 50:
            return colors.blue
        else:
            return colors.lightblue

    def draw_centered_string(x, y, text, font_size, font_type="Helvetica-Bold", color=colors.black):
        canvas.setFillColor(color)
        canvas.setFont(font_type, font_size)
        text_width = canvas.stringWidth(text, font_type, font_size)
        canvas.drawString(x - text_width / 2, y, text)

    def draw_description(x_left, start_y, description, max_width):
        words = description.split(' ')
        new_line = '\t'  
        y = start_y - 50  
        canvas.setFont("Helvetica", score_desc_font_size)  
        canvas.setFillColor(colors.black)

        for word in words:
            if canvas.stringWidth(new_line + word + ' ', "Helvetica", score_desc_font_size) > max_width:
                canvas.drawString(x_left, y, new_line.strip())
                y -= score_desc_line_height
                new_line = '\t' + word + ' ' 
            else:
                new_line += word + ' '
        if new_line:
            canvas.drawString(x_left, y, new_line.strip())

    score_separation = 230  # Horizontal space between scores
    x_center_sentiment = (start_x + end_x) / 2 - score_separation / 2
    x_center_approval = (start_x + end_x) / 2 + score_separation / 2

    description_start_y = start_y - 5  # Starting y position for descriptions

    # Overall Sentiment Index
    OSI_description = ("Represents the average sentiment tone across all reviews. "
                       "Calculated using the TextBlob library.")
    draw_centered_string(x_center_sentiment, start_y, "Overall Sentiment Index", score_name_font_size, color = colors.black)
    canvas.setFillColor(colors.black)  # Reset color to black for the score value and descriptions
    draw_centered_string(x_center_sentiment, start_y - 30, f"{sentiment_index:.2f}%", score_value_font_size, color=get_score_color(sentiment_index))
    draw_description(x_center_sentiment - score_separation / 2 + 25, description_start_y, OSI_description, score_separation - 20)

    # Sentiment Approval Rating
    SAR_description = ("Represents the percentage of reviews that are positive. "
                       "Calculated using Hugging Face transformers sentiment analysis pipeline.")
    draw_centered_string(x_center_approval, start_y, "Sentiment Approval Rating", score_name_font_size, color = colors.black)
    canvas.setFillColor(colors.black)  # Reset color to black again
    draw_centered_string(x_center_approval, start_y - 30, f"{approval_rating:.2f}%", score_value_font_size, color=get_score_color(approval_rating))
    draw_description(x_center_approval - score_separation / 2 + 25, description_start_y, SAR_description, score_separation - 20)

# Adds product information at top of report
def add_product_info_section(canvas, product_details, start_y, width):
    font_type = "Helvetica-Bold"
    font_size = 12
    color = Color(0.2, 0.2, 0.3)

    line_height = 15
    decoration_height = 5

    # Calculate positions
    text_width_title = canvas.stringWidth(product_title, font_type, font_size)

    product_title = product_details["data"]["product_title"]
    product_price = product_details["data"]["product_price"]

    if product_price == '':
        product_price = product_details["data"]["product_original_price"]

    text_width_price = canvas.stringWidth("Price: " + product_price, font_type, font_size)
    start_x_title = (width - text_width_title) / 2
    start_x_price = (width - text_width_price) / 2

    # Draw product title
    canvas.setFont(font_type, font_size)
    canvas.setFillColor(color)
    canvas.drawString(start_x_title, start_y, product_title)

    # Draw product price
    canvas.drawString(start_x_price, start_y - line_height, "Price: " + product_price)

# Adds titles for charts
def add_chart_title(canvas, title, x, y, chart_width,font):
    canvas.setFont("Helvetica-Bold", font)  # Set font and size for the title
    title_width = canvas.stringWidth(title, "Helvetica-Bold", font)
    title_x = x + (chart_width - title_width) / 2  # Center the title
    canvas.drawString(title_x, y, title)

# Adds rating distribution chart to report
def add_rating_distribution_chart(canvas, x, y):
    ratio = 1.33
    chart_height = 150  # Height of the chart
    chart_width = chart_height * ratio  # Width of the chart calculated based on the aspect ratio
    image_path = 'rating_distribution_chart.png'

    # Add title above the chart
    chart_title = "Rating Distribution Chart"
    title_offset_y = 10  # Reduced vertical offset for the title above the chart
    add_chart_title(canvas, chart_title, x, y + chart_height + title_offset_y, chart_width,font=11)

    # Draw the chart
    canvas.drawImage(image_path, x, y, chart_width, chart_height, mask='auto')

# Adds verified purchases chart to report
def add_verified_purchases_chart(canvas, x, y):
    ratio = 1.33
    chart_height = 150
    chart_width = chart_height * ratio
    image_path = 'verified_purchases_chart.png'

    # Add title above the chart
    chart_title = "Verified Purchases Chart"
    title_offset_y = 10  # Reduced vertical offset for the title above the chart
    add_chart_title(canvas, chart_title, x, y + chart_height + title_offset_y, chart_width,font=11)

    # Draw the chart
    canvas.drawImage(image_path, x, y, chart_width, chart_height, mask='auto')

# Adds ratings over time chart to report
def add_ratings_over_time_chart(canvas, x, y):
    chart_height = 120  # Height of the chart
    chart_width =  300 # Width of the chart calculated based on the aspect ratio
    image_path = 'ratings_over_time_chart.png'

    # Add title above the chart
    chart_title = "Ratings Over Time"
    title_offset_y = 8  # Vertical offset for the title above the chart
    add_chart_title(canvas, chart_title, x, y + chart_height + title_offset_y, chart_width, font = 10)

    # Draw the chart
    canvas.drawImage(image_path, x, y, chart_width, chart_height, mask='auto')


# Adds additional product information to report
def add_additional_information_section(canvas, product_details, start_x, start_y):
    title = "Additional Information"
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(start_x, start_y, title)

    # Adjust start position for bullet points
    bullet_start_y = start_y - 20
    bullet_indent = 10
    text_indent = 20
    line_height = 14

    canvas.setFont("Helvetica", 10)

    # Bullet point for Sales Volume
    sales_volume = product_details['data']['sales_volume']
    canvas.drawString(start_x + bullet_indent, bullet_start_y, "-")
    sales_volume_text = f"Sales Volume: {sales_volume}"
    canvas.drawString(start_x + text_indent, bullet_start_y, sales_volume_text)
    bullet_start_y -= line_height

    # Bullet point for Climate Pledge Friendly
    climate_pledge_friendly = product_details['data']['climate_pledge_friendly']
    canvas.drawString(start_x + bullet_indent, bullet_start_y, "-")
    climate_pledge_text = "Climate Pledge Friendly: " + ("Yes" if climate_pledge_friendly else "No")
    canvas.drawString(start_x + text_indent, bullet_start_y, climate_pledge_text)
    bullet_start_y -= line_height

    # Bullet point for Amazon Choice
    is_amazon_choice = product_details['data']['is_amazon_choice']
    canvas.drawString(start_x + bullet_indent, bullet_start_y, "-")
    amazon_choice_text = "Amazon Choice: " + ("Yes" if is_amazon_choice  else "No")
    canvas.drawString(start_x + text_indent, bullet_start_y, amazon_choice_text)

# Call functions to create report some customizable dimension aspects. 

pdf_path = "product_report.pdf"
c = create_pdf(pdf_path)


add_product_info_section(c, product_details, 530, landscape(letter)[0])
add_takeaways_section(c, final_condensed_takeaways, 50, 415, 300)
add_scores_section(c, Overall_Sentiment_Index, Sentiment_Approval_Rating, 390, 470, 700)

add_rating_distribution_chart(c, 330, 160) 
add_verified_purchases_chart(c, 570, 160) 
add_ratings_over_time_chart(c, 400, 20)
add_additional_information_section(c, product_details,50,100)

c.save()