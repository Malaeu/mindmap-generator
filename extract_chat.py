#!/usr/bin/env python3
"""Extract chat conversation from HTML and convert to Markdown for mindmap generation."""

import html

from bs4 import BeautifulSoup


def extract_chat_to_markdown(html_file: str, output_file: str):
    """Extract chat messages from HTML and convert to clean Markdown."""
    
    with open(html_file, encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Extract title
    title = soup.find('div', class_='conversation-title')
    title_text = title.text.strip() if title else "Chat Conversation"
    
    # Extract all messages
    messages = soup.find_all('div', class_='message')
    
    markdown_content = f"# {title_text}\n\n"
    
    for msg in messages:
        # Get sender and timestamp
        sender = msg.find('span', class_='sender')
        timestamp = msg.find('span', class_='timestamp')
        content_div = msg.find('div', class_='message-content')
        
        if sender and content_div:
            sender_text = sender.text.strip()
            timestamp_text = timestamp.text.strip() if timestamp else ""
            
            markdown_content += f"## {sender_text} - {timestamp_text}\n\n"
            
            # Process content - extract text, preserving code blocks
            for elem in content_div.children:
                if elem.name == 'p':
                    # Regular paragraph
                    text = elem.get_text(strip=True)
                    text = html.unescape(text)  # Decode HTML entities
                    markdown_content += f"{text}\n\n"
                    
                elif elem.name == 'pre':
                    # Code block
                    code = elem.get_text(strip=False)
                    code = html.unescape(code)
                    # Check if it has a language class
                    lang_class = elem.get('class', [])
                    lang = ''
                    if lang_class and 'code-block' in lang_class[0]:
                        # Try to extract language from class
                        for cls in lang_class:
                            if cls != 'code-block':
                                lang = cls
                                break
                    markdown_content += f"```{lang}\n{code}\n```\n\n"
                    
                elif elem.name == 'ul':
                    # Bullet list
                    for li in elem.find_all('li'):
                        text = li.get_text(strip=True)
                        text = html.unescape(text)
                        markdown_content += f"- {text}\n"
                    markdown_content += "\n"
                    
                elif elem.name == 'ol':
                    # Numbered list
                    for i, li in enumerate(elem.find_all('li'), 1):
                        text = li.get_text(strip=True)
                        text = html.unescape(text)
                        markdown_content += f"{i}. {text}\n"
                    markdown_content += "\n"
            
            markdown_content += "---\n\n"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return markdown_content

# Extract the conversation
input_file = "/home/chirurgie/Downloads/chat_conversations_2025-08-28T07-16-14.766Z_Exploring_20Mathematical_20Infinity.html"
output_file = "/media/chirurgie/hdd01/Soft/GitHub/mindmap-generator/infinity_conversation.md"

content = extract_chat_to_markdown(input_file, output_file)
print(f"Extracted conversation to {output_file}")
print(f"Content length: {len(content)} characters")