from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd

class Crawl:
    def __init__(self, name,link) -> None:
        self.driver = webdriver.Chrome()#tạo đối tượng drive
        self.name_file = name+".csv"
        #lấy link :
        self.driver.get(link)
        # các attributes cần lấy
        self.attribute = {
            'user' : 'a-profile-name',
            'sumary': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold',
            'star':'review-star-rating',#data-hook
            'country&time':'a-size-base a-color-secondary review-date',
            'kind item': 'a-row a-spacing-mini review-data review-format-strip',
            'comment':'a-row a-spacing-small review-data',
            'image':'review-image-tile-section',
            'translateEnglish': 'a-button a-button-base'
        }
        
    def get_content_page(self, star, header):
        # dịch sang tiếng anh
        try:
            button = self.driver.find_element(By.CSS_SELECTOR, '[data-hook = "cr-translate-these-reviews-link"]')
            webdriver.ActionChains(self.driver).move_to_element(button).click(button).perform()
            time.sleep(7)
        except Exception:
            pass


        dict_content = {
            'user':[],
            'star':[],
            'sumary':[],
            'country&time':[],
            'kind item':[],
            'image':[],
            'comment':[]
            }
        if header:
            df = pd.DataFrame(dict_content)
            df.to_csv(self.name_file, header=True, index = False,mode='a')

        contents = self.driver.find_elements(By.CSS_SELECTOR, '[class = "a-section review aok-relative"]')
        
        for content in contents:
            dict_content['user'].append(content.find_element(By.CSS_SELECTOR, f'[class = "a-profile-name"]').text)

            dict_content['star'].append(star)

            dict_content['sumary'].append(content.find_element(
                    By.CSS_SELECTOR, '[class = "a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold"]'
                ).text)

            dict_content['country&time'].append(content.find_element(By.CSS_SELECTOR, f'[class = "a-size-base a-color-secondary review-date"]').text)

            dict_content['kind item'].append(content.find_element(By.CSS_SELECTOR, f'[class = "a-row a-spacing-mini review-data review-format-strip"]').text)

            try:
                content.find_element(By.CSS_SELECTOR, f'[class = "review-image-tile-section"]').text
                dict_content['image'].append(1)
            except Exception:
                dict_content['image'].append(0)

            dict_content['comment'].append(content.find_element(By.CSS_SELECTOR, f'[class = "a-row a-spacing-small review-data"]').text)
        df = pd.DataFrame(dict_content)
        df.to_csv(self.name_file, header=False, index = False,mode='a')
        time.sleep(1)
    def crawl(self):
        # tìm button next page
        stars = [5,4,3,2,1]
        header = True
        for star in stars:
            button_start = self.driver.find_element(By.CSS_SELECTOR, f'[class = "a-link-normal {star}star"]')
            # button_start = driver.find_element(By.CSS_SELECTOR, f'[class = "a-link-normal 5star"]')
            webdriver.ActionChains(self.driver).move_to_element(button_start).click(button_start).perform()
            time.sleep(7)
            
            while(True):
                try:
                    # lấy dữ liệu ở đây
                    
                    self.get_content_page(star, header)
                    header = False

                    button = self.driver.find_element(By.CSS_SELECTOR, '[class = "a-last"]')
                    webdriver.ActionChains(self.driver).move_to_element(button).click(button).perform()
                    time.sleep(7)
                except Exception:
                    break

if __name__ == "__main__":
    # go = Crawl("Razer DeathAdder Essential Gaming Mouse") # đã crawl
    link = "https://www.amazon.com/SteelSeries-Arctis-Multi-Platform-Gaming-Headset/product-reviews/B09ZWMYHCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    go = Crawl('SteelSeries New Arctis Nova 3 Multi-Platform Gaming Headset', link)
    go.crawl()
