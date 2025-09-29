import asyncio
import winsdk.windows.devices.geolocation as wdg
import requests

class LocationService:
    @staticmethod
    async def get_coords():
        locator = wdg.Geolocator()
        pos = await locator.get_geoposition_async()
        return [pos.coordinate.latitude, pos.coordinate.longitude]

    def get_location(self):
        try:
            coords = asyncio.run(self.get_coords())
            
            # Get English address
            geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={coords[0]}&lon={coords[1]}&accept-language=en"
            headers = {'User-Agent': 'MyApp/1.0'}
            geocode_response = requests.get(geocode_url, headers=headers)
            geocode_data = geocode_response.json()
            address_en = geocode_data.get("display_name", "Address not found")

            # Get Arabic address
            geocode_url_arabic = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={coords[0]}&lon={coords[1]}&accept-language=ar"
            geocode_response_arabic = requests.get(geocode_url_arabic, headers=headers)
            geocode_data_arabic = geocode_response_arabic.json()
            address_ar = geocode_data_arabic.get("display_name", "عنوان غير موجود")

            return {
                'coordinates': coords,
                'address_en': address_en,
                'address_ar': address_ar
            }

        except PermissionError:
            print("ERROR: Location access denied. Please follow these steps:")
            print("1. Open Windows Settings")
            print("2. Go to Privacy & Security > Location")
            print("3. Enable 'Location services'")
            print("4. Enable 'Let apps access your location'")
            print("5. Enable 'Let desktop apps access your location'")
            return None

if __name__ == "__main__":
    # Test the location service
    loc_service = LocationService()
    location_info = loc_service.get_location()
    
    if location_info:
        print("\n=== Location Information ===")
        print(f"📍 Coordinates:")
        print(f"   Latitude:  {location_info['coordinates'][0]}")
        print(f"   Longitude: {location_info['coordinates'][1]}")
        print(f"\n📮 Address (English):")
        print(f"   {location_info['address_en']}")
        print(f"\n📮 Address (Arabic):")
        print(f"   {location_info['address_ar']}")
        print("========================")