from pathlib import Path
import urllib.request
import urllib
import imghdr
import posixpath
import re
import numpy as np 
import cv2
import sys
'''
Python api to download image form Bing.
Author: Guru Prasad (g.gaurav541@gmail.com)
'''


class Bing:
    def __init__(self, query,limit, output_dir, adult, timeout,  filter='', verbose=True,):
        self.download_count = 0
        self.classes = query if  isinstance(query, list) else [query]
        self.query=0
        self.xquery=self.query
        self.breaker=0
        self.output_dir = output_dir
        self.adult = adult
        self.filter = filter
        self.verbose = verbose
        self.seen = set()
        assert type(limit) == int, "limit must be integer"
        self.limit = limit
        assert type(timeout) == int, "timeout must be integer"
        self.timeout = timeout

        # self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        self.page_counter = 0
        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
      'AppleWebKit/537.11 (KHTML, like Gecko) '
      'Chrome/23.0.1271.64 Safari/537.11',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
      'Accept-Encoding': 'none',
      'Accept-Language': 'en-US,en;q=0.8',
      'Connection': 'keep-alive'}


    def get_filter(self, shorthand):
            if shorthand == "line" or shorthand == "linedrawing":
                return "+filterui:photo-linedrawing"
            elif shorthand == "photo":
                return "+filterui:photo-photo"
            elif shorthand == "clipart":
                return "+filterui:photo-clipart"
            elif shorthand == "gif" or shorthand == "animatedgif":
                return "+filterui:photo-animatedgif"
            elif shorthand == "transparent":
                return "+filterui:photo-transparent"
            else:
                return ""


    def save_image(self, link, file_path):
        request = urllib.request.Request(link, None, self.headers)
        image = urllib.request.urlopen(request, timeout=self.timeout).read()
        if not imghdr.what(None, image):
            print('[Error]Invalid image, not saving {}\n'.format(link))
            raise ValueError('Invalid image, not saving {}\n'.format(link))
        ############################
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img=cv2.resize(img,(600,600))
        wname="{} || Press 'd'-download|'s'-skip".format(self.query)
        cv2.imshow(wname,img)
        while True:
            key = cv2.waitKey(0) 
            if key == ord('d'):
                with open(str(file_path), 'wb') as f:
                    f.write(image)
                    self.download_count += 1
                break
            if key == ord('s'):
                break   
            if key==ord('q'):
                cv2.destroyAllWindows()
                extra=input("Enter extra query: ")
                self.xquery=self.query+" "+extra
                self.breaker=1
                break  
            if key==ord("x"):
                cv2.destroyAllWindows()
                sys.exit()
            if key==ord("n"):
                cv2.destroyAllWindows()
                self.breaker=2
                break
        if self.download_count==self.limit:
            cv2.destroyAllWindows()
        #########################     
        #with open(str(file_path), 'wb') as f:
         #   f.write(image)

    
    def download_image(self, link):
        #self.download_count += 1
        # Get the image link
        try:
            path = urllib.parse.urlsplit(link).path
            filename = posixpath.basename(path).split('?')[0]
            file_type = filename.split(".")[-1]
            if file_type.lower() not in ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
                file_type = "jpg"
                
            if self.verbose:
                # Download the image
                print("[%] Downloading Image #{} from {}".format(self.download_count, link))
                
            self.save_image(link, self.output_dir.joinpath("{}_{}.{}".format(self.query,
                str(self.download_count), file_type)))
            if self.verbose:
                print("[%] File Downloaded !\n")

        except Exception as e:
            #self.download_count -= 1
            print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))

    
    def run(self):
        for q in self.classes:
            self.query=q
            self.xquery=self.query+" plants"
            while self.download_count < self.limit:
                self.breaker=0
                if self.verbose:
                    print('\n\n[!!]Indexing page: {}\n'.format(self.page_counter + 1))
                # Parse the page source and download pics
                request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(self.xquery) \
                            + '&first=' + str(self.page_counter) + '&count=' + str(self.limit) \
                            + '&adlt=' + self.adult + '&qft=' + ('' if self.filter is None else self.get_filter(self.filter))
                request = urllib.request.Request(request_url, None, headers=self.headers)
                response = urllib.request.urlopen(request)
                html = response.read().decode('utf8')
                if html ==  "":
                    print("[%] No more images are available")
                    break
                links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
                if self.verbose:
                    print("[%] Indexed {} Images on Page {}.".format(len(links), self.page_counter + 1))
                    print("\n===============================================\n")

                for link in links:
                    if self.download_count < self.limit and link not in self.seen:
                        self.seen.add(link)
                        self.download_image(link)
                        if self.breaker==1 or self.breaker==2:
                            break

                if self.breaker==2:
                    break            
                self.page_counter += 1
            self.download_count=0
            print("\n\n[%] Done. Downloaded {} images.".format(self.download_count))
