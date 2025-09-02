antes de hacerle el refactor al lab tenia funcionando el frontend-backend-lab para hacer inferencias simples, es decir hacia el prompt en un input del frontend y este iba al backend y el backend levantaba un LLM local mediante el lab, y la respuesta del lab llegaba correctamente al frontend, luego refactorice el lab y agregue un pequeño flujo de nodos: Creator > Challenger > Refiner, pero para probar esto ultimo ya no lo hice desde el frontend, sino desde el mismo lab a traves de un test, pero funciono, el prompt inicial paso por los 3 nodos, o mejor dicho por los 3 models. Luego refactorice un poco del frontend para permitir un flujo de 3 nodos ya que solo estaba configurado para una inferencia simple. Pero ahi quedo



 Ahora analiza que tan preparado esta este lab y que tan     │
│   dificil seria adaptar un sistema que deje elegir varias     │
│   personalizaciones:\                                         │
│   \                                                           │
│   Si bien un orquestador y la creacion de RAGs son cosas      │
│   relacionadas pero que no se entrecruzan tanto, me gustaria  │
│   poder elegir entre:\                                        │
│   A)Modelo de embedding como bge-m3 u otro. \                 │
│   B)Base de vectores como Milvus, Weaviate o Pinecone\        │
│   C)Otras herramientas que luego vaya agregando deben ser     │
│   personalizable\                                             │
│   \                                                           │
│   El frontend mandaria que herramienta o funcion selecciono   │
│   el usuario y cada flujo seria diferente, muchas mas         │
│   posibilidades 