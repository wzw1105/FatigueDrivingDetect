import requests
import re
def getOutterIP():
    ip = ''
    try:
        print("getting ip...")
        res = requests.get('https://myip.ipip.net', timeout=2).text
        ip = re.findall(r'(\d+\.\d+\.\d+\.\d+)', res)
        ip = ip[0] if ip else ''
    except:
        print("Error in getting IP")
    return ip

def get_location(ip_address):
    url = f"https://ipinfo.io/{ip_address}/json"
    response = requests.get(url)
    data = response.json()
    print(data)
    print(response.status_code)
    return data

def get_gaode_location(ip_address):
    ret = requests.get(f"https://restapi.amap.com/v3/ip?ip={ip_address}&output=json&key=7adc97f7624ef1865767693e1b183478")
    print(ret.json())


if __name__ == "__main__":
    ip_address = getOutterIP()  # 将其替换为您要查询的实际IP地址
    #location_info = get_location(ip_address)
    get_gaode_location(ip_address)
    # print("IP地址:", location_info.get("ip"))
    # print("位置:", location_info.get("city"))
    # print("地区:", location_info.get("region"))
    # print("国家:", location_info.get("country"))
    # print("经纬度:", location_info.get("loc"))