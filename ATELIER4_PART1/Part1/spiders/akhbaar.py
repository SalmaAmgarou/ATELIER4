import scrapy

from ..items import Part1Item


class AkhbaarSpider(scrapy.Spider):
    name = "akhbaar"
    allowed_domains = ["www.bbc.com"]
    start_urls = ["https://www.bbc.com/arabic/topics/c719d2el19nt"]

    def parse(self, response):
        articles = response.css('li.bbc-t44f9r')
        for article in articles:
            article_url = article.css('a.e1d658bg0::attr(href)').get()
            yield response.follow(article_url, callback =self.parse_article_page)
        # Next page links generated dynamically
        base_url = "https://www.bbc.com/arabic/topics/c719d2el19nt?page="
        page_numbers = range(1, 50)  # Change the range according to your needs
        next_page_links = [base_url + str(page) for page in page_numbers]

        for link in next_page_links:
            yield response.follow(link, callback=self.parse)
    def parse_article_page(self, response):
        content_paragraphs = response.css("main.bbc-fa0wmp").xpath('string()').getall()
        item = Part1Item()
        item['text'] = content_paragraphs
        yield item
