import urllib.request
import requests
import re
from bs4 import BeautifulSoup

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}
# url = 'http://www.csrc.gov.cn/pub/newsite/zjhxwfb/xwdd/'
# res = requests.get(url, headers=headers)
# res.encoding = 'utf-8'
# soup = BeautifulSoup(res.text, "lxml")
#
# news_list = soup.find('div', {'class': 'fl_list'})
#
# print(news_list)
#
# for each in news_list.find_all(href=re.compile("html")):
#     url2 = ''.join(["http://www.csrc.gov.cn/pub/newsite/zjhxwfb/xwdd", each["href"]])
#     res2 = requests.get(url2, headers=headers)
#     res2.encoding = 'utf-8'
#     soup2 = BeautifulSoup(res2.text, "lxml")
#     news = soup2.find('div', {'class': 'in_main'})
#     news_title = []
#     news_time = []
#     news_content = []
#
#     title = news.find('div', {'class': 'title'}).get_text().strip()
#     time = news.find('div', {'class': 'time'}).get_text().strip()
#     content = news.find('div', {'class': 'Custom_UnionStyle'}).get_text().strip()
#     news_title.append(title)
#     news_time.append(time)
#     news_content.append(content)
#     print('新闻标题: ', title)
#     print(time)
#     print('内容： ', content, '\n', '\n')

def getMainPage(url):
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, "lxml")

    temp = soup.find('div', {'class': 'in_list gao_1'})

    new_urls = []
    #print(temp)
    new_urls.append(temp.div.a.attrs['href'])
    for line in temp.find_all('li'):
        new_urls.append(line.a.attrs['href'])

    new_urls = [url + new[2:] for new in new_urls]

    return new_urls

def getNewsCont(url):
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, "lxml")
    news = soup.find('div', {'class': 'content'})

    title = news.find('div', {'class': 'title'}).get_text().strip()
    time = news.find('div', {'class': 'time'}).get_text().strip()
    content = news.find('div', {'class': 'Custom_UnionStyle'}).get_text().strip()

    new = {}
    new['title'] = title
    new['time'] = time
    new['content'] = content
    return new


def main():

    mainpage = 'http://www.csrc.gov.cn/pub/newsite/'
    #获得主页和所需板块的各个新闻标题的url
    urls = getMainPage(mainpage)
    print(urls)
    #对每个页面的内容进行爬取
    news = []
    for url in urls:
        news.append(getNewsCont(url))
    #显示内容
    for new in news:
        print(new)
    #持久化存储
    with open('./news.txt','wb') as f:
        f.write(news)

if __name__ == "__main__":
    main()