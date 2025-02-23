# SignLingo - học ngôn ngữ ký hiệu trực quan

### Prerequisites

**Node version 14.x**

### Cloning the repository

```shell
git clone https://github.com/AntonioErdeljac/next14-duolingo-clone.git
```

### Install packages

```shell
npm i
```

### Setup .env file


```js
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=""
CLERK_SECRET_KEY=""
DATABASE_URL="postgresql://..."
STRIPE_API_KEY=""
NEXT_PUBLIC_APP_URL="http://localhost:3000"
STRIPE_WEBHOOK_SECRET=""
```

### Setup Drizzle ORM

```shell
npm run db:push

```

### Seed the app

```shell
npm run db:seed

```

or

```shell
npm run db:prod

```
### Start FastAPI server
Requires Python 3.10. Conda virtual enviroment is recommended
```shell
cd 'backend'
pip install -r requirements.txt
fastapi dev main.py
```

### Start the app

```shell
npm run dev
```
