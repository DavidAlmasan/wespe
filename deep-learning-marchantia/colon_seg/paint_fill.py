from PIL import Image, ImageDraw

def open_image(path):
    newImage = Image.open(path)
    rgbImage = Image.new("RGB", newImage.size)
    rgbImage.paste(newImage)
    return rgbImage

def save_image(image, path):
    image.save(path, 'png')

def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel

def floodfill_queue(image, filled_pixel_map, x, y, oldColor, newColor):  
    width, height = image.size

    if  get_pixel(image, x, y) != oldColor or filled_pixel_map[x, y] != oldColor:  
        return  
  
    q = [(x, y)]  
    while q:  
        n = q.pop(0)  
        filled_pixel_map[n[0], n[1]] = newColor
        for node in [(n[0]+1, n[1]), (n[0]-1, n[1]), (n[0], n[1]+1), (n[0], n[1]-1)]:  
            if (node[0] >= width or node[0] < 0 or node[1] >= height or node[1] < 0 ) == False:
                if filled_pixel_map[node[0], node[1]] == oldColor:  
                    filled_pixel_map[node[0], node[1]] = newColor  
                    q.append(node)  
        
image_path = './data/marchantia_gt/g2_t017_c001.png'

image = open_image(image_path)
filled_image = open_image(image_path)

pixel_map = filled_image.load()

width, height = image.size

oldColor = (0, 0, 0) # black
newColor = (0, 255, 0) # green
borderColor = (255, 255, 225) # white

# for i in range(width):
#     for j in range(height):
#         floodfill_queue(image, pixel_map, i, j, oldColor, newColor)
      
floodfill_queue(image, pixel_map, 500, 500, oldColor, newColor)

save_image(filled_image, './filled.png')


