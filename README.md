# Amazon Product Sentiment Report Generator

The goal of this project was to speed up, automate, and improve the decision making process of purchasing a product online. 

## Overview
This tool efficiently processes and analyzes customer reviews from Amazon products. It leverages state of the art sentiment analysis and data processing techniques to parse through vast amounts of text, extracting the most pertinent information and compiling it into an easily digestible format.

## How It Works
Inputting an Amazon product link into the Python script initiates a process of fetching and analyzing customer reviews, which then creates an informative and detailed report. This report showcases key insights, sentiment scores, and trends drawn from the review data.

## Example Report
See below an example of the report generated from a product link:

<a href="https://github.com/bencoleman24/AmazonSentimentalReports/assets/97268624/c86abd3e-4b04-4195-ae8c-c4bdc89dc483">
    <img src="https://github.com/bencoleman24/AmazonSentimentalReports/assets/97268624/c86abd3e-4b04-4195-ae8c-c4bdc89dc483" width="550" alt="Report Screenshot"/>
</a>

## Features
- **Advanced Sentiment Analysis**: Utilizes NLP techniques to determine the sentiment of user reviews, gathering key takeaways and numeric sentiment score values. 
- **Insightful Data Visualization**: Generates clear, intuitive charts and graphs to represent the data, making it easy to understand the overall sentiment trends.
- **Comprehensive Reporting**: Produces detailed PDF reports summarizing the findings, complete with visual aids and key takeaways.

## Usage
To generate a report, follow these steps:
1. Set up your .env file in the project root with your RapidAPI and OpenAI API keys
2. Run the script
3. Enter the Amazon product URL when prompted
4. Enjoy the report pdf file in working directory

## Limitations and Scope

- Due to rate limits for api it currently processes up to 10 pages of reviews (can be modified).
- This tool is made specifically for Amazon products but could be extrapolated to other E-commerce websites with altered api utilization.
- Best used for products with a substancial amount of reviews.
- Models can inaccurately classify the sentiment of reviews. Scores are insightful but not consistently accurate.

### Tools/Technologies

#### Report Generation

- Matplotlib & Seaborn: Used to create charts and graphs.
- ReportLab: Used for generating the final PDF report. Allowed for creating complex layouts, embedding visualization, statistics, and text.

#### Analysis
- Hugging Face Transformers: The primary tool for sentiment analysis, known for their state-of-the-art performance in NLP tasks. Integrated into the script to analyze individual reviews and classify their sentiment.
- TextBlob: Supplementary tool for aggregate sentiment analysis. Provides a polarity score representing the overall sentiment tone of a text - from negative to positive. Used for calculating the Overall Sentiment Index by analyzing the combined text of all reviews for a product.
- OpenAI API: Leveraged for additional NLP tasks, such as summarizing key points from reviews.













