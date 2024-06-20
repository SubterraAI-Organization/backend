FROM node:21.5.0-alpine AS build
WORKDIR /app

ARG VITE_BACKEND_URL
ARG VITE_AUTHORIZATION
ARG VITE_CSRFTOKEN

ENV VITE_BACKEND_URL=$VITE_BACKEND_URL
ENV VITE_AUTHORIZATION=$VITE_AUTHORIZATION
ENV VITE_CSRFTOKEN=$VITE_CSRFTOKEN

COPY ["package.json", "./"]

RUN npm install

COPY . .

RUN npx vite build


FROM node:21.5.0-alpine AS production
WORKDIR /app

COPY --from=build /app/dist ./dist

RUN npm install -g serve

EXPOSE 3000/tcp

CMD ["npx", "serve", "-s", "dist"]
