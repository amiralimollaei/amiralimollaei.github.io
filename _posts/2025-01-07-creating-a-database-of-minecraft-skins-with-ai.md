---
title: Creating a Dataset of Minecraft Skins Using AI
description: >-
  I passed 197k Minecraft skins to a VLM to create one of the most accurate datasets of labelled minecraft skins available
author: amiralimollaei
date: 2025-01-07 18:00:00 +0330
categories: []
tags: []
pin: false
# media_subpath:
---

I passed 197k minecraft skins to a VLM (Vision Language Model) to describe them in varying levels of detail, this dataset has various AI applications that I'll mention later.

## Problem 1: AI Does Not Understand Skin Files
If you upload a Minecraft skin to chatgpt and ask for a description of it, it probably wouldn't do it very well.
> This is because chatgpt can't understand the 3D nature of Minecraft skins and it's pixel art nature, It is seen as a very blury image with a black background.

But if you render the minecraft skin, then the AI can understand and describe it significantly better, because now it's a 3D object and not a blury mess.

For the rendering I used a python library called `minepi` and modified the code to get better lighting and better performance.

## Problem 2: Formulated Descriptions Ruin Creativity
Suppose we have a minecraft skin of Albert Einstein, would you describe Albert Einstein as being an old man with gray hair and wearing a lab coat or do you simply say it's Albert Einstein? 

Below is the prompt I used for describing a minecraft character:
```
Describe this 3D Minecraft skin using the format: "A Minecraft skin of [(a/an) sth. or someone] with [sth.] wearing [sth.], [extras]"

Examples:
- "a creeper with red eyes wearing red pants, creeper (mob)"
- "Albert Einstein wearing a lab coat"
- "a girl wearing a pig costume, festival themed"
- "Mikel Jackson wearing a black shirt, singer, stage outfit"
- "a skeleton, skeleton (mob)"
- "a man wearing a rainbow outfit"
- "a man with blue hair wearing red pants, no shirt, and black gloves"
- "an explorer, wearing an explorer outfit"
- "a pig wearing a bee costume, striped yellow shirt, bee themed"
- "Batman with a mask"
```

But there's a problem, if all the descriptions follow the same formula, that means to use an AI trained on this dataset, we have to follow that formula too, otherwise the results wouldn't be great.

For this I passed the descriptions again to another LLM to paraphrase them using the following prompt:
```
You are a perceptual sentence paraphrase and summarization agent:

a good paraphrase should be adequate and fluent while being as different as possible on the surface lexical form. With respect to this definition, the 3 key metrics that measures the quality of paraphrases are
Adequacy (Is the meaning preserved adequately?)
Fluency (Is the paraphrase fluent English?)
Diversity (Lexical / Phrasal / Syntactical) (How much has the paraphrase changed the original sentence?)

The generated text should convey the same meaning as the original context (Adequacy).
The generated text should be fluent / grammatically correct (Fluency).

ONLY OUTPUT A JSON containing 5 sentences, ordered from no perceptual detail (a few words) to high detail (full paragraph), every iteration increases the perceptual detail, with the following format:

{"paraphrases": ["...", ... ]}
```

## Problem 3: The Huge Amount Of Computation
To process 197 thousand images is not an easy task, it requires a LOT of computation, something that I Just couldn't do on my PC.

So, I used [Deepinfra](https://deepinfra.com/) a cloud computing service that allows 200 concurrent requests to their APIs, and was able to use their APIs to do the entire processing on their servers using `Llama3.2-90B-Vision-Instruct` and `Llama3.3-70B-Instruct`, the code I wrote achived this in 3 hours, with 128 concurrent requests maintained at all time.

## The Dataset
Below is some samples from the dataset,
It's accurate around %90 of the time, and in my opinion this is one of the most accurate datasets in this field.

![Sample 1](/assets/posts/1/sample1.png)
![Sample 2](/assets/posts/1/sample2.png)
![Sample 3](/assets/posts/1/sample3.png)
![Sample 4](/assets/posts/1/sample4.png)

---

### What can this dataset be used for?
This dataset can be used for training AI models that can generate or modify minecraft skins based on a prompt, similar to image generator websites, or can be used to train AI models that categorise minecraft skins (e.g. detecting inappropriate or harmful stuff) or finding similaritis/variants of existing minecraft skins, or creating AI models that can describe any given Minecraft skin for a fraction of the computation that was previously possible.
