---
name: x-post
description: A post for X (Twitter), with or without a picture, or a thread; also adapting an Instagram draft for X.
---
1. Write the text first: on X people read before they look. One idea, the strongest sentence
   first, no filler. The limit is 280 characters including spaces and links; aim for 120 to 240.
   At most one or two hashtags, and only if they add reach (inside the sentence is fine). Plain
   text, no Markdown.
2. create_draft with platform x and the text as the caption. To adapt an Instagram draft, use
   from_draft with its number and rewrite the caption for X: shorter, sharper, no hashtag block.
   The pictures and texts come along, re-cropped to 16:9. Look at where the texts landed and fix
   them with edit_text if they no longer fit. X takes at most 4 pictures: pass media with the ones
   to keep if the Instagram draft had more.
3. A picture is optional: add one (pictures) when it shows what words cannot (the product, a
   result, a chart). 16:9 is the default shape; 1:1 or 4:5 also work on X (aspect).
4. For a thread, make one draft per post, titled "Thread 1/4", "Thread 2/4" and so on, each
   under 280 characters. The first post must work on its own.
5. Check the character count in the result, then say it is ready.
