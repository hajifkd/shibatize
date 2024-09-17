import os
from openai import AsyncAzureOpenAI
import asyncio
import aiohttp
from bs4 import BeautifulSoup


async def main():
    arxiv_id = '2407.19755'
    author_name = 'ほげ田'
    author_affiliation = 'ほげ大学'

    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://arxiv.org/abs/{arxiv_id}') as response:
            if response.status == 200:
                data = await response.text()
                soup = BeautifulSoup(data, 'html.parser')
                abstract = soup.find('blockquote', class_='abstract mathjax')
                title = soup.find('h1', class_='title mathjax')
            else:
                print(f'Failed to download abstracts from {response.url}')
                return

    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.
    deployment_name = 'gpt-4o'

    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""研究室で{author_affiliation}の{author_name}さんに、論文「{title.text}」に関してセミナー講演を依頼するにあたって、大学の事務に理由書を提出する必要があります。その理由書を作成してください。
理由書は必ず日本語でお願いします。英語を含めてはいけません。ネガティブな内容は一切入れないでください。
また、理由書の分量は、**一段落**、**1000文字程度**で書いてください。
特に、{author_name}さんは学生であるため、以下の点を含めてください。ただし、明示的に書くのではなく、自然に含めてください。
* その学生の指導教員ではなく、その学生ではないといけない理由。ただし、他の著者を貶めるような内容は含めないでください。
* 論文の要約
また、次の点に気を付ける必要があります。
* 「指導講演」、という言葉は使ってはいけません。使いたければ、代わりに「指導助言」と言い換えてください。
* {author_name}さんにしか依頼できない理由を強めに記載してください。以前、「良い模範となると判断した為」と書いたところ、これは高い単価を設定する理由には弱いといわれました。
* なるべく一瞥しただけではわかりづらい内容で書いてください。
論文のアブストラクトは以下の通りです。
{abstract.text}
"""},
        ]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
