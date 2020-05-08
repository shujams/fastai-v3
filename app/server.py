import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.googleapis.com/drive/v3/files/1h6azhw9HhSMHgi6s07mC3szuYi-XIuT2?alt=media&key=AIzaSyACjIUZH-eCqYWZ4ZptnKbTZs9HMQhd3AE'
export_file_name = 'COVID-CT-densenet121-4.pkl'

classes = ['2019-nCoV-Negative', '2019-nCoV-Positive', 'Severe', 'Asymptomatic', 'Mild', 'Critical', 'Absorption stage',
       'Consolidation stage', 'Early stage', 'Dissipation stage',
       'Moderate', 'Pregnant', 'Male', 'Female', '75.0 y/o', '76.0 y/o', '70.0 y/o', '73.0 y/o', '44.0 y/o',
       '65.0 y/o', '37.0 y/o', '50.0 y/o', '1.0 y/o', '33.0 y/o',
       '21.0 y/o', '69.0 y/o', '57.0 y/o', '64.0 y/o', '60.0 y/o',
       '72.0 y/o', '63.0 y/o', '36.0 y/o', '34.0 y/o', '48.0 y/o',
       '45.0 y/o', '39.0 y/o', '66.0 y/o', '41.0 y/o', '40.0 y/o',
       '32.0 y/o', '49.0 y/o', '23.0 y/o', '71.0 y/o', '46.0 y/o',
       '27.0 y/o', '28.0 y/o', '31.0 y/o', '59.0 y/o', '62.0 y/o',
       '55.0 y/o', '"Diamond Princess" Cruise Ship', 'Beijing, China',
       'Changsha, China', 'China (Unspecified Region)',
       'Guangdon, China', 'Hainan, China', 'Hubei, China', 'Hunan, China',
       'Jingmen, Hubei, China', 'Qingdao, China', 'Shanghai, China',
       'Shenzhen, China', 'Sichuan, China', 'Wuhan, China',
       "Xi'an, China", 'Zhejiang, China']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
