# GPT-2 Transformer Build Video Transcript

## Future-session context

- **Video:** [Building GPT-2 from scratch, explained line by line](https://www.youtube.com/watch?v=CfJ3Cxtlcps)
- **Transcript source:** User-provided YouTube transcript pasted into this session.
- **Project described:** Implement a GPT-2-style decoder-only Transformer in PyTorch, covering configuration, token and positional embeddings, causal self-attention, multi-head attention, MLPs, residual connections, layer normalization, language-model output head, next-token training/loss, and autoregressive generation.
- **Transcript quality:** Preserved as supplied. It appears to be auto-generated and contains speech-recognition errors, malformed timestamps, and occasional duplicated lines; check code names, formulas, and numerical examples against the video or a reliable reference while implementing.
- **Use in future sessions:** Treat this file as source context for building the project. Begin with a small configurable model and validate tensor shapes and masking before attempting GPT-2-scale settings.

## Transcript (as supplied)

Chapter 1: Introduction
0:000 secondsYou've used chat GPT. You type a question and  a perfect answer just appears. It feels like  
0:077 secondsmagic. But what if I told you the entire secret  to that magic is this? Now, don't worry. I know  
0:1515 secondsthis looks like a wall of code. This is the  complete GPT2. And right here, this first part,  
0:2323 secondsthis is the causal self-attention block. It's the  absolute heart of GPT. We're going to break this  
0:3030 secondsdown completely. And the next piece is the MLP and  the transformer block. This is what processes the  
0:3838 secondsinformation from the attention mechanism. It's  the thinking part of the network. And finally,  
0:4444 secondsyou put it all together into the full GPT2 model  class. It's just the embeddings, those transformer  
0:5151 secondsblocks we saw, and an output layer. That's it.  These roughly 100 lines of Python contain all  
1:001 minutethe core architectural ideas behind multi-billion  dollar models like GPT. There is no hidden code,  
1:081 minute, 8 secondsno secret sauce. My promise to you is  simple. Give me 100 minutes and I will  
1:151 minute, 15 secondsmake the transformer click for you forever. We  are going to walk through this file line by line.  
1:241 minute, 24 secondsYou will understand not just what each line  does but why it's there. That is the key. The  
1:321 minute, 32 secondsmagic will dissolve into understandable, elegant  engineering. This diagram is our road map. Look at  
1:401 minute, 40 secondsit here. We're going to see this a few times. We  will start at the very beginning with embeddings  
1:471 minute, 47 secondswhich turn words into vectors. You can see them  highlighted in green at the bottom. Then we'll  
1:531 minute, 53 secondsbuild the core engine, the transformer block. You  see it here in orange. This is where the real work  
2:012 minutes, 1 secondhappens. Finally, we'll build the output layer  that makes the prediction. That's the purple block  
2:072 minutes, 7 secondsright at the top. The most important component is  causal self attention. It's all governed by this  
2:142 minutes, 14 secondsone formula. A tension of q kv equals softmax of  qkrpose over the square root of d subk plus a mask  
2:252 minutes, 25 secondsm all times v. I know this looks intimidating  now. I totally get it. Trust me, by the end of  
2:322 minutes, 32 secondsthis video it will be second nature. Every project  starts with a blueprint. For us that's the GPT  
Chapter 2: The GPT Config
2:412 minutes, 41 secondsconfig. It's a simple data class holding all our  parameters. We have vocab size, block underscore  
2:492 minutes, 49 secondssize, n underscore layer, n underscore head, n  underscore ebd, and dropout. These are the knobs  
2:582 minutes, 58 secondsyou turn to make the model bigger or smaller.  Let's look at this table together. We're going to  
3:043 minutes, 4 secondswalk through this step by step. First up, we have  vocab size. This controls the vocabulary size,  
3:113 minutes, 11 secondsor how many unique words the model knows.  For the real GPT2 small, this was 50,257.
3:203 minutes, 20 secondsNext, block size. This is the context  window. It's how far back the model can  
3:263 minutes, 26 secondssee at once. GPT2 could see 1,024 tokens  into the past. Then N layer. This is the  
3:353 minutes, 35 secondsmodel depth. It's how many of those transformer  blocks are stacked on top of each other. More  
3:413 minutes, 41 secondslayers means a more powerful model. The small  version had 12. After that comes n head. This  
3:493 minutes, 49 secondsis the model limb commute. It controls how many  parallel conversations the attention mechanism  
3:553 minutes, 55 secondscan have at once. We'll dive deep into this.  GPT2 also used 12 for this. And finally, N_EMD,  
4:064 minutes, 6 secondsthe embedding dimension. This is the size or  richness of the vectors that represent each token.  
4:134 minutes, 13 secondsGPT2 used a size of 768. So by just changing  these numbers, the exact same 100 lines of  
4:234 minutes, 23 secondscode can create a tiny toy model or the full  scale GPT2. It's incredibly flexible. With  
4:314 minutes, 31 secondsour blueprint in hand, we're ready to lay the  first brick. Next, we will build the embedding  
4:374 minutes, 37 secondslayers that turn symbol numbers into the rich  vectors our model needs. Let's get our hands  
Chapter 3: Token Embeddings
4:444 minutes, 44 secondsdirty. Our journey starts with the very first  functional layer of the model. Let's look at  
4:504 minutes, 50 secondsour code blueprint and find exactly what we're  building. Look here on the screen. We have our  
4:574 minutes, 57 secondsclass gpt2 which inherits from nnn.module.  Inside the init method, we are building this  
5:065 minutes, 6 secondsline right now. The one that says self.wte equals  nn.bedding with config.vocab size and config.n_d.
5:215 minutes, 21 secondsThis is the word token embedding. The next  line wp. Well, that's the next chapter.   So why do we need this?
5:345 minutes, 34 secondsWhat problem are we actually solving? Well, here's  the problem. Raw token ids. The input to our model  
5:435 minutes, 43 secondsis a sequence of numbers like an array of 521  108. But these are just meaningless IDs from  
5:535 minutes, 53 secondsa vocabulary list. A neural net can't do math  on them. It just doesn't work that way. Think  
6:006 minutesabout it. The number 21 isn't 4.2 times more word  than the number five. That's absurd. The distance  
6:116 minutes, 11 secondsbetween them is completely meaningless. So, how  do we give these numbers meaning? Our goal is to  
6:196 minutes, 19 secondsmap each token ID to a vector. A vector is just  a list of numbers representing a coordinate in  
6:286 minutes, 28 secondsspace. And here's the key idea. The location  of these coordinates should actually mean  
6:366 minutes, 36 secondssomething. Let me give you an analogy. Let's  think about representing colors. A bad way,  
6:436 minutes, 43 secondsjust like our token IDs, would be to assign them  arbitrary numbers. Maybe a dictionary where red  
6:506 minutes, 50 secondsis one, orange is two, and blue is eight. A  computer sees eight and one but has no idea  
6:586 minutes, 58 secondsthat orange is supposed to be between red and  blue. The numbers are completely arbitrary. Now,  
7:067 minutes, 6 secondsa good way would be to use vectors. We can  represent colors in a coordinate system.  
7:127 minutes, 12 secondsLet's say the x-axis is redness and the y-axis is  bluess. In this system, red could be the vector  
7:207 minutes, 20 seconds0.9 0.1. Orange could be 0.8 0.2. See, it's  super close to red. And blue could be 0.1,
7:377 minutes, 37 secondswhich is way far away from red. Now, the  distance between points is meaningful.  
7:427 minutes, 42 secondsWe're going to do the exact same thing for words,  but in a space with 768 dimensions. The mechanism  
7:557 minutes, 55 secondsfor this is the n.bedding layer in PyTorch. It's  nothing more than a giant lookup table. You can  
8:038 minutes, 3 secondsthink of it as a learnable coordinate book.  Let's build one and look inside. We'll import  
8:088 minutes, 8 secondstorch and torch.n. We'll set up a tiny config  for our example. Let's say vocab size is 10 and  
8:178 minutes, 17 secondsn_omd is 3. That's the number of dimensions  in our semantic space. The layer itself,  
8:248 minutes, 24 secondswhich we'll call token_bedding table, is just  nn.bing with vocab size and n_md. the book itself.  
8:368 minutes, 36 secondsThe actual coordinates are in theweight attribute.  Each row is a words coordinate. Let's print the  
8:448 minutes, 44 secondsshape of our coordinate book. And then let's print  the actual content. And here's the output. Check  
8:528 minutes, 52 secondsit out. The shape of our coordinate book is torch  size with 10 rows and three columns. Makes sense.  
9:009 minutesAnd the content of the book is a tensor filled  with initially random coordinates. You can see  
9:069 minutes, 6 secondsthe first row is the coordinate for token zero.  The second row is the coordinate for token 1 and  
9:129 minutes, 12 secondsso on all the way down to token 9. And notice  it says requires grad equals true. This is the  
9:209 minutes, 20 secondscore. It's really just a matrix of shape vocab  size by n_bd. That's all it is. And did you see  
9:319 minutes, 31 secondsthat requires grad equals true part? That's the  magic. It means during training, the model will  
9:419 minutes, 41 secondslearn the best possible coordinates for each word  to help it make better predictions. The model  
9:499 minutes, 49 secondsitself will figure out what redness and bless  mean for words. We've successfully transformed  
9:569 minutes, 56 secondsour data. Our input of meaningless integers  like 251 becomes an output of rich 3D embedding  
10:0510 minutes, 5 secondsvectors. Each token now has a rich 3D semantic  representation. So what have we gained? We now  
10:1610 minutes, 16 secondshave a representation where words with similar  meanings will end up close to each other in this  
10:2210 minutes, 22 secondshighdimensional space. This is what enables the  famous vector math analogy. You've probably heard  
10:2810 minutes, 28 secondsof the vector for king minus the vector for man  plus the vector for woman is approximately equal  
10:3610 minutes, 36 secondsto the vector for queen. It's incredible. But  this representation has one massive flaw. It  
10:4810 minutes, 48 secondsis shredded and contextfree. The vector for the  word bank is identical in these two sentences.  
10:5810 minutes, 58 secondsNumber one, I sat on the riverbank. Number two, I  withdrew money from the bank. The embedding layer  
11:0611 minutes, 6 secondshas no idea which bank we're talking about. It  just gives back the same vector every time. This  
11:1411 minutes, 14 secondsis the central problem the transformer is designed  to solve. How can a token's vector be dynamically  
11:2211 minutes, 22 secondsadjusted based on its context? To even begin to  answer that, we first need to give the model a  
Chapter 4: Positional Embeddings
11:3011 minutes, 30 secondssense of order. Let's locate our next component  in the code blueprint. We've built the word token  
11:3911 minutes, 39 secondsembedding the WTE. Now we are building this line  right now self.WP WP which is another innind.  
11:5011 minutes, 50 secondslayer. This is for the word positional embedding.  So how do we do it? The solution used in GPT is  
11:5911 minutes, 59 secondswonderfully simple and elegant. Think about this.  Just as we learned a unique vector for each word,  
12:0812 minutes, 8 secondswe will now learn a unique vector for each  position. It's that simple. Each position gets  
12:1712 minutes, 17 secondsits own unique vector. So there's a vector  that literally means I am at position one,  
12:2412 minutes, 24 secondsanother for I am at position two and so on  all the way up to our maximum sequence length,  
12:3012 minutes, 30 secondsthe block size. And how do we implement this? It's  another in embedding layer. It's a lookup table,  
12:3812 minutes, 38 secondsbut this time for positions, not for words. Now,  you're probably asking the key question. Why  
12:4612 minutes, 46 secondsdoes adding the position vector to the word vector  work? Won't that just scramble all the information  
12:5212 minutes, 52 secondswe so carefully created? Here's the key insight.  It works for a few reasons. First, they exist in  
13:0113 minutes, 1 secondthe same space. Both the word vectors and position  vectors live in that same 768dimensional space.  
13:1113 minutes, 11 secondsSecond, adding them creates a unique point  in that space. The model learns to create  
13:1813 minutes, 18 secondspositional vectors such that adding them to a word  vector creates a new distinct representation. So  
13:2613 minutes, 26 secondsthe word 'the' at position 5 is now a completely  different point from the word 'the' at position  
13:3313 minutes, 33 seconds20. The network learns to understand  this distinction. Let's see this in code.  
13:4013 minutes, 40 secondsIt's always the best way to make it click. Okay,  let's walk through this together line by line.  
13:4713 minutes, 47 secondsFirst, our tiny config. B, T, and C are 2, 5, and  three. That's our batch, time or sequence length,  
13:5813 minutes, 58 secondsand channels or embedding dimension. Our vocab  size is 10. And our block size, our model's  
14:0614 minutes, 6 secondsmaximum sequence length, is 8. Now for the layers.  First we have the same token embedding table from  
14:1314 minutes, 13 secondsthe last section. Nothing new there. And now the  new one, the position embedding table. Notice  
14:2214 minutes, 22 secondsits vocabulary is the block size. It has eight  possible positions it can represent. Okay. Next  
14:2914 minutes, 29 secondsup, the input data. We'll create a random tensor  of token IDs called idx with a shape of 2x5.  
14:3914 minutes, 39 secondsAll right, step one, get the token embeddings.  Just as before, we pass our idx into the token  
14:4614 minutes, 46 secondsembedding table. The result talk worship has a  shape of 2x 5x3. Now for step two, the new part,  
14:5614 minutes, 56 secondsgetting the positional embeddings. We only need  the positions for our current sequence length t,  
15:0315 minutes, 3 secondswhich is five. So we use torch.arrange a range  from 0 to t, which gives us a tensor of 0 1 2 3  
15:1015 minutes, 10 seconds4. We then look these position ids up in our new  position embedding table. The result pause embed  
15:1815 minutes, 18 secondshas a shape of 5x3. And for step three, we combine  them. It's just simple addition. X equals talkb  
15:2815 minutes, 28 secondsplus pause embeds. Finally, let's print the shapes  and see what happened. And here's the output. The  
15:3715 minutes, 37 secondsshape of our token embeddings is torch size of  253. The shape of our positional embeddings is 53.  
15:4515 minutes, 45 secondsAnd the shape of the final combined embeddings is  253. Now look at those shapes again. How can you  
15:5315 minutes, 53 secondsadd a 2x 5x3 tensor to a 5x3 tensor? The answer  is PyTorch uses broadcasting. PyTorch is smart.  
16:0316 minutes, 3 secondsIt automatically takes that 5x3 position tensor  and adds it to each of the two sequences in our  
16:1016 minutes, 10 secondsbatch. It just works. It's incredibly efficient.  This also answers another key question. What if  
16:1716 minutes, 17 secondsour sequence length is less than the block size?  This is the common case. Our block size is 8,  
16:2416 minutes, 24 secondsbut our current sequence length t is only five.  The solution is that smart line of code torch a  
16:3016 minutes, 30 secondsrange from 0 to t. It only generates the positions  for the actual sequence length we're processing  
16:3616 minutes, 36 secondsright now. The result, perfect automatic handling.  It just works. We have now prepared the input for  
16:4316 minutes, 43 secondsthe main event. Our tensor X contains vectors  that are now aware of both a token's identity  
16:5016 minutes, 50 secondsand its absolute position in the sequence. This  is the input that will flow into the stack of  
16:5716 minutes, 57 secondstransformer blocks. It's time to build the heart  of the machine. We've laid the foundation. Every  
17:0417 minutes, 4 secondsword has a vector. Every position has a  vector. But we still have the riverbank  
17:1017 minutes, 10 secondsversus money bank problem. Our model has no way  to adjust a word's meaning based on its context.
Chapter 5: Self-Attention Intuition
17:1917 minutes, 19 secondsThis is the heart of the transformer.
17:2517 minutes, 25 secondsTo make this crystal clear, let's forget code for  a moment and focus on one single ambiguous word.  
17:3317 minutes, 33 secondsCrane. First sentence, the crane ate a fish. Here,  crane is obviously a bird. Second sentence, the  
17:4417 minutes, 44 secondscrane lifted the steel. Here, crane is a machine.  Right now the starting vector for crane is  
17:5317 minutes, 53 secondsidentical in both sentences. We need a mechanism  to update this vector based on its neighbors. That  
18:0118 minutes, 1 secondmechanism is called self attention. And this is  its formula. Attention of qkv equals softmax of  
18:0918 minutes, 9 secondsqranspose over the square root of d subk all time  v. H I know it looks complex but by the end of  
18:1918 minutes, 19 secondsthis section you'll understand it intuitively.  The entire process happens in three steps. One  
18:2718 minutes, 27 secondsscoring, two normalizing and three aggregating.  But first what are Q, K and V? They are three  
18:3718 minutes, 37 secondsdistinct vectors for every single word created by  projecting the input vector X. The query or Q is  
18:4518 minutes, 45 secondsthe word's search query. It's what it's looking  for. The key or K is the word's label or keyword.  
18:5418 minutes, 54 secondsIt's what it is. And the value or V is the  word's payload, the information it offers. Now,  
19:0119 minutes, 1 secondyou might be asking a brilliant question. Why do  we need a separate value vector? Isn't our input  
19:0719 minutes, 7 secondsvector X already the information? The V vector is  a transformed version of X packaged for others to  
19:1419 minutes, 14 secondsconsume. Look at this table. The input vector  X is like your entire knowledge and resume.
19:2319 minutes, 23 secondsIt's the raw complete information. But the  value vector V, that's your prepared elevator  
19:2919 minutes, 29 secondspitch. It's packaged consumable information  that's ready to be aggregated. It's made from  
19:3519 minutes, 35 secondsa learned transformation of X. The model learns  the best elevator pitch for each word. Okay,  
19:4219 minutes, 42 secondsanalogy over. Let's see the numbers. We'll use a  dead simple 2D space. Dimension one represents is  
19:5019 minutes, 50 secondsit an animal and dimension two represents is it a  machine. Here are the vectors for our words. Check  
19:5919 minutes, 59 secondsout this table. Notice the crane vectors are all  ambiguous like 7.7. But ate and fish are high on  
20:0820 minutes, 8 secondsthe animal dimension and lifted and steel are high  on the machine dimension. All right, sentence one.  
20:1520 minutes, 15 secondsThe crane ate a fish. We're going to walk through  this step by step. First up, scoring. The crane's  
20:2320 minutes, 23 secondsquery, which is 0.70.7, is going to probe every  other word's key. Let's do the math. The score  
20:3120 minutes, 31 secondsfor crane looking at itself is 0.70.7 dotted  with 0.70.7 which equals 0.98. That's a high  
20:4220 minutes, 42 secondsself-affinity. Now the score for crane looking at  8 is 0.70.7 dotted with 0.90.1. That equals 0.70.  
20:5620 minutes, 56 secondsA high match. And crane looking at fish. It's  0.70.7 dotted with 0.80.2 which also equals 0.70.  
21:1021 minutes, 10 secondsAnother high match. Step two, normalizing. We take  those raw scores 0.98, 0.70, 0.70 and we run them  
21:2121 minutes, 21 secondsthrough softmax to convert them into percentages.  The result, our attention weights are 0.4  
21:2821 minutes, 28 seconds4 0.3 0.3 This is the aha moment. Crane has  decided it will listen 40% to its original self,  
21:3921 minutes, 39 seconds30% to 8, and 30% to fish. Okay, step three,  aggregating. We take those weights and multiply  
21:4821 minutes, 48 secondsthem by the value vectors. So the new vector for  crane is 0.4 4 * the value of crane which is 050.5
21:5821 minutes, 58 secondsplus 0.3 * the value of 8  0.90.1 plus 0.3 * the value  
22:0622 minutes, 6 secondsof fish 0.80.2 that gives us 0.20.2 plus 0.270.03
22:1622 minutes, 16 secondsplus 0.24.06. The new vector for crane is now 0.71  0.29. It is heavily skewed towards dimension one  
22:2922 minutes, 29 secondsanimal. The ambiguity is resolved. Now let's do  sentence two. The crane lifted the steel. The  
22:3722 minutes, 37 secondsprocess is the same. So let's move a bit faster.  First scoring. The crane's query is the exact same  
22:4322 minutes, 43 secondsas before. It probes the new words. And look at  this. The scores are identical. 0.98 0.70 0.70. So  
22:5422 minutes, 54 secondsfor step two, normalizing the attention weights  are the same. 0.4 0.3 0.3. Here's the kicker.  
23:0223 minutes, 2 secondsThey're applied to different context words. So for  our final step, aggregating, we take our weights  
23:0823 minutes, 8 secondsand apply them to the new value vectors. It's  0.4 * the crane value plus 0.3 * the lifted value  
23:1623 minutes, 16 secondsplus 0.3 * the steel value. This becomes 0.20.2  + 0.030.27 plus 0.06 0.24. And there it is. The  
23:3123 minutes, 31 secondsnew vector is 0.29 0.71. Now heavily skewed  towards dimension 2 machine. The exact same  
23:4223 minutes, 42 secondsinitial crane vector has been transformed into two  completely different contextaware vectors. This is  
23:5223 minutes, 52 secondsthe power of self attention. It's a communication  mechanism that allows every word to dynamically  
23:5923 minutes, 59 secondspull in information from its context to refine  its own meaning. We've built the intuition.  
Chapter 6: Attention Implementation
24:0824 minutes, 8 secondsWe saw how the word crane could resolve its  ambiguity by listening to its neighbors. Now  
24:1424 minutes, 14 secondsit's time to translate that exact process  into the language of linear algebra. Fast,  
24:2224 minutes, 22 secondsefficient matrix operations. Our map for this  section is the formula itself. Look at this on  
24:3024 minutes, 30 secondsyour screen. We are going to build this step by  step. Attention of Q, K, and V equals soft max of  
24:4224 minutes, 42 secondsQ * Krpose divided by the square root of D subK.  All of that time V. I know, I know it might look  
24:5424 minutes, 54 secondsa little intimidating, but we are going to build  it piece by piece and it'll make perfect sense.  
25:0125 minutes, 1 secondFirst, we will build this with raw tensors to  see every single number. Then we will encapsulate  
25:0725 minutes, 7 secondsthat logic into a clean, reusable PyTorch node.  Part one, the raw tensor walkthrough. Let's use  
25:1625 minutes, 16 secondsa simple four-word sentence. So, t= 4. We'll  use a tiny embedding dimension c= 2. And we'll  
25:2525 minutes, 25 secondsprocess one sentence at a time. So, b equals 1.  Okay, let's import torch torch.n and a few other  
25:3525 minutes, 35 secondsthings we'll need. We set bc to 1 4 and two. Now,  here's a pretend input tensor from our embedding  
25:4725 minutes, 47 secondslayers for the sentence, a crane ate fish.  We'll call it x. It's just a torch.tensor with  
25:5525 minutes, 55 secondsthe values you see here on the screen. Step one,  projecting X into Q, K, and V. To get our query,  
26:0626 minutes, 6 secondskey, and value vectors, we use learnable  linear transformations. These are just  
26:1326 minutes, 13 secondsstandard PyTorchn.linear layers. So, we'll define  a QC_RJ, a K underscorep, and a V underscorep.
26:2526 minutes, 25 secondsNow, just to make this tutorial  reproducible, we'll manually set  
26:2926 minutes, 29 secondsthe weights using torch.mmanual seed42. This  way, you can get the exact same numbers I do.  
26:3826 minutes, 38 secondsAnd now we perform the projections. Q= Q ofX,  K= K_ra ofX, and V= V_ra of X. Simple as that.  
26:5526 minutes, 55 secondsLet's track our tensor shapes. This is the  most important habit to build. Seriously,  
27:0327 minutes, 3 secondslook at this table. Our variable X has the shape  1 42. That's our batch of raw input vectors. After  
27:1127 minutes, 11 secondsthe projection, Q, K, and V all have the exact  same shape, 1 4 2. They represent the query,  
27:2127 minutes, 21 secondsthe key, and the value vector for each  of our four tokens. Okay. Step two,  
27:2727 minutes, 27 secondscalculate attention scores. This is the core of  the communication. We compute the dotproduct of  
27:3427 minutes, 34 secondsevery token's query with every other token's key.  So Q has a shape of 1 42. To multiply it with K,  
27:4427 minutes, 44 secondswe need to transpose K's last two dimensions.  So K.Transpose transpose of -2, -1 gives us a  
27:5327 minutes, 53 secondsnew shape of 1 2 4. The resulting shape, a 1x4x4  matrix. Here's the code. Scores equals Q at symbol  
28:0428 minutes, 4 secondsk.transpose -2, -1. And here are the raw scores,  the attention matrix. It's a tensor of size 1 44.
28:2028 minutes, 20 secondsLet's look at this closely. The first row  shows the scores for A looking at every  
28:2628 minutes, 26 secondsother word. The second row is for crane looking  at every other word. And look at that second  
28:3328 minutes, 33 secondsrow. It has the highest score 1.3888 with itself.  This makes perfect sense. A word is usually most  
28:4328 minutes, 43 secondsrelated to itself. Step three and four. scale  and soft max. We scale for numerical stability.  
28:5228 minutes, 52 secondsThen we use softmax to turn each row into weights  that sum to one. So first we get d subk which is  
29:0129 minutes, 1 secondjust the size of the last dimension of k. Then  scaled scores equals scores divided by math do  
29:0929 minutes, 9 secondssquare root of d subk. Finally attention weights  equals f.oftmax softmax of the scaled scores along  
29:1829 minutes, 18 secondsthe last dimension. Step five, aggregate the  values. This is it. Now we use our calculated  
29:2629 minutes, 26 secondsweights to create a weighted average of the value  vectors. Our attention underscore weights tensor  
29:3429 minutes, 34 secondshas a shape of 144. Our V tensor has a shape of  142. The resulting shape is 142. Exactly what we  
29:4629 minutes, 46 secondsstarted with. The code is simple. Output equals  attention_weights at symbol v. And here's our  
29:5529 minutes, 55 secondsfinal output. A tensor of size 1 42 containing  our new contextaware vectors. [Music] Here is  
30:0630 minutes, 6 secondsa summary of the entire process. Look at this  table. Step one, we create Q, K, and V. Step two,  
30:1530 minutes, 15 secondswe multiply Q and K transpose to get raw scores.  Step three, we scale by the square root of D subK.  
30:2430 minutes, 24 secondsStep four, we apply soft max to get attention  probabilities. And step five, we multiply those  
30:3130 minutes, 31 secondsprobabilities by V to get our final contextaware  output vectors. Success. We've transformed our  
30:3830 minutes, 38 secondsinput X into an output of the exact same shape  where each token's vector is now contextaware.  
30:4830 minutes, 48 secondsPart two, encapsulating the logic in an NN module.  The raw tensor walkthrough is fantastic for  
30:5630 minutes, 56 secondsunderstanding, but in practice, we package this  logic into a reusable class. This makes our code  
31:0331 minutes, 3 secondsclean, organized, and easy to stack. Here is the  final reusable module. By the end of this section,  
31:1231 minutes, 12 secondsevery single line will be crystal clear. We  define a class single head self attention  
31:2031 minutes, 20 secondsthat inherits from nn.module. Inside the init, we  define one linear layer. And in the forward pass,  
31:2931 minutes, 29 secondswe do all five steps. Let's look at the  underscore_init method. the fused linear  
31:3731 minutes, 37 secondslayer. Instead of three separate linear layers,  we use one larger, much faster one. The manual  
31:4531 minutes, 45 secondswalkthrough on the left is conceptually clear,  but the fuse layer on the right is computationally  
31:5131 minutes, 51 secondsefficient. We define a single layer C_ATR that  goes from our embedding dimension to three times  
31:5931 minutes, 59 secondsthe embedding dimension. Now the forward method  projection and splitting. First we get qv equals  
32:0932 minutes, 9 secondsself c_atn of x. This produces a big tensor of  shape bt 3 * c. Then we do something clever. Q kv  
32:2332 minutes, 23 secondsequals qkv.split with c along dimension 2. This  carves that big tensor up into three chunks of  
32:3232 minutes, 32 secondssize C. This gives us back our separate Q, K, and  V tensors, each with the shape BTC. The rest of  
32:4132 minutes, 41 secondsthe code is a direct one:one implementation of the  formula we just walked through. It's identical. It  
32:4932 minutes, 49 secondsworks perfectly. We have successfully implemented  the core of self attention and formalized it in a  
32:5532 minutes, 55 secondsclean, professional module. However, our model  has a flaw, a big one. For language generation,  
Chapter 7: Causal Masking
33:0733 minutes, 7 secondstokens can see into the future. In our  current setup, the word a can see crane,  
33:1433 minutes, 14 secondseight ate, and fish. When we're generating  text one word at a time, that's cheating.
33:2433 minutes, 24 secondsLet's think about what we're trying to do. We  want to build an auto reggressive model. It's  
33:2933 minutes, 29 secondsa fancy term for a model that generates text one  token at a time. When predicting the next word  
33:3633 minutes, 36 secondsfor a crane 8, its decision must be based only  on tokens it has already seen. A crane 8. It  
33:4733 minutes, 47 secondscannot be allowed to see the actual answer. Fish.  But look at the attention matrix we calculated.  
33:5633 minutes, 56 secondsCheck this out. The first row for A is attending  to all four tokens. That's cheating. The second  
34:0534 minutes, 5 secondsrow for crane also attends to all four tokens.  This is a huge problem. A token at position T  
34:1334 minutes, 13 secondsmust only be allowed to communicate with tokens  from the past. From positions zero up to T.  
34:2234 minutes, 22 secondsIt cannot see the future. The solution is as  elegant as it is effective. The causal mask. We  
34:3134 minutes, 31 secondswill modify the attention scores before softmax.  We mask out future positions by setting their  
34:3934 minutes, 39 secondsscores to negative infinity. Now why negative  infinity? This is the key technical trick. The  
34:4834 minutes, 48 secondssoftmax function involves an exponential. You see  the formula here. It's e to the power of x sub i  
34:5434 minutes, 54 seconds/ the sum of all the e to the power of x's.  So what is the exponential of infinity? e to  
35:0335 minutes, 3 secondsthe power of negative infinity is approximately  zero. By using negative infinity, we force the  
35:1135 minutes, 11 secondsattention weights for any future tokens to become  zero. It's brilliant. Let's see this in action.  
35:1935 minutes, 19 secondsThis is the visual aha moment. We'll start with  the setup raw scores. We pick up with our scaled  
35:2635 minutes, 26 secondsscores matrix from earlier. You remember this  one, right? The shape is 1 144. I'll just put the  
35:3235 minutes, 32 secondstensor up on the screen here. Step one, create  the mask. A lower triangular matrix is perfect  
35:3935 minutes, 39 secondsfor this. We can get one with torch.trill. Ones  are positions we keep and zeros are the future  
35:4535 minutes, 45 secondspositions we're going to block. So we set t equals  4 and our mask is torch.trill of torch1s of t t.  
35:5335 minutes, 53 secondsAnd here's the mask it creates. Look at this.  Row zero for a can only see column zero a makes  
36:0036 minutessense. Row one for crane can see columns zero  and one. So a and itself. The zeros are what  
36:0836 minutes, 8 secondsblock those future connections. Step two, apply  the mask. We replace all values in scaled scores  
36:1436 minutes, 14 secondswith negative infinity wherever our mask is equal  to zero. The code is masked scores equals scaled  
36:2136 minutes, 21 secondsscores.mmasked fill where mask equals equals 0  with float of negative infinity. And here are  
36:2736 minutes, 27 secondsthe scores after masking. Look at that beautiful  upper right triangle. It's all negative infinity.
36:3636 minutes, 36 secondsPerfect. All future positions  have been obliterated.   Step three, rerun softmax. Now let's apply softmax  to these masked scores and watch the magic happen.  
36:4836 minutes, 48 secondsAttention major equals fs softmax of masked scores  along the last dimension. And here are our final  
36:5536 minutes, 55 secondscausal attention weights. This is it. The upper  right triangle is now all zer. A only attends  
37:0237 minutes, 2 secondsto itself 100%. Crane attends to A and itself.  Eight attends to a crane and eight and so on.  
37:1237 minutes, 12 secondsWe have fixed the time traveling bug. Let's look  at this table for the unmasked attention. Could  
37:1837 minutes, 18 secondscrane attend to fish? Yes. Could eight attend  to fish? Yes. But now with causal attention,  
37:2737 minutes, 27 secondsthe answer to both is no. 0%. Problem solved.  Part two. Encapsulating in the inend module.  
37:3737 minutes, 37 secondsLet's look at the init method and a cool function  called register buffer. We need to store our mask  
37:4337 minutes, 43 secondsas part of the module, but it's not a trainable  parameter. For this, we use register buffer. So,  
37:4937 minutes, 49 secondsinside our class causal self attention, we call  self.register buffer. We'll name it bias and  
37:5537 minutes, 55 secondswe give it our torch.trill mask. So, quick quiz.  What's a buffer? A buffer is a tensor that is part  
38:0338 minutes, 3 secondsof the model state. It gets saved with the model's  state [ __ ] It moves to the GPU with 2 device,  
38:0938 minutes, 9 secondsbut it is not a model parameter. So, it is  not updated by the optimizer. It's perfect  
38:1538 minutes, 15 secondsfor our fixed causal mask. Now, for the forward  method, the forward pass is updated with one  
38:2138 minutes, 21 secondscrucial line right before the softmax. After we  calculate our scaled scores, we add the new line.  
38:2738 minutes, 27 secondsScaled scores equals scaled scores.mmasked  fill. Using our stored buffer self.bias,  
38:3238 minutes, 32 secondsbias. We slice it to match the current sequence  length t. And again, wherever it's zero, we fill  
38:3938 minutes, 39 secondswith negative infinity. Then we just run softmax  like before. We have now built a fully functional  
38:4738 minutes, 47 secondscausal single-headed attention mechanism. It  can learn context, but it can no longer cheat.
38:5638 minutes, 56 secondsBut it's still just a single attention mechanism.  It's like one person in a meeting trying to track  
39:0339 minutes, 3 secondseverything at once. Grammar, meaning, long range  context, all at the same time. That's a lot of  
Chapter 8: Multi-Head Attention
39:1139 minutes, 11 secondspressure. To make it truly powerful, we need to  allow it to have many conversations at the same  
39:1739 minutes, 17 secondstime. So, what's the solution? You hire a team of  specialists. This is the core idea of multi- head  
39:2739 minutes, 27 secondsattention. Instead of one overworked mechanism, we  have several working in parallel. Head one might  
39:3639 minutes, 36 secondsbecome a specialist in syntax, you know, verb  object relationships. Head two might specialize  
39:4339 minutes, 43 secondsin semantics like related meanings. Head three  might track pronoun references and so on. The  
39:5139 minutes, 51 secondsentire process follows a simple narrative. Look  at this diagram. It's just split, then attend,  
40:0040 minutesthen merge. That's it. Part one, the raw tensor  walkthrough. Let's use the actual dimensions from  
40:0940 minutes, 9 secondsthe GPT2 small model to make this real. Our  embedding dimension C is 768. The number of  
40:1840 minutes, 18 secondsheads N head is 12. And that means the dimension  per head dim is c / n head which is 64. Okay,  
40:3140 minutes, 31 secondslet's walk through this together step by step.  Step one, splitting c into n head and head dim.  
40:4240 minutes, 42 secondsWe'll start with a dummy Q tensor with a realistic  shape. Let's say Q equals torch.rand randen with B  
40:5040 minutes, 50 secondsT C. So a batch of one, a sequence length of four,  and our embedding dimension of 768. First, let's  
41:0041 minuteslook at its starting shape. We print the original  Q shape, and we get torch size of 1 4 768. Makes  
41:1041 minutes, 10 secondssense. Now for step 1.2, the dot view operation.  Here's where we carve up that last dimension.  
41:1941 minutes, 19 secondsThe code is QC reshaped equals Q dot view  with B T N head and head dim. And look at  
41:2941 minutes, 29 secondsthe new shape after dot view. It's torch size  of 1 4 1264. We've successfully created our  
41:3741 minutes, 37 secondshead dimension. But we're not done. Step 1.3,  thetranspose operation. This is so important.  
41:4741 minutes, 47 secondsWe need to bring the number of heads  dimension to the front. This isolates  
41:5241 minutes, 52 secondsthe heads so PyTorch can process them all in  parallel. The code is Q_fal Q_res.transpose.  
42:0142 minutes, 1 secondSwitching dimensions 1 and two. So the  final reshaped Q shape is now 1 12464.
42:1042 minutes, 10 secondsThe key insight is in that final transpose.  We move the n head dimension to the front. So,  
42:1742 minutes, 17 secondsPyTorch's broadcasting will treat it  like a new batch dimension. All our   attention math will now happen  12 times at once independently.
42:2842 minutes, 28 secondsStep two, run attention in parallel. We do  the exact same reshaping for K and V. So K  
42:3642 minutes, 36 secondsunderscorefinal and V underscorefinal both get  reshaped and transposed into that same 112464  
42:4342 minutes, 43 secondsshape. Now for the attention calculation which  is happening 12 times at once. We multiply Q_fal  
42:5242 minutes, 52 secondsby K_fal transposed. That's a B NHTD matrix  multiplied by a BNHDT which gives us a BNHT.  
43:0343 minutes, 3 secondsThen we scale it. we apply softmax and finally  we multiply the attention weights by v_fal. So a  
43:1243 minutes, 12 secondsbnhtt matrix times a bnhthd gives us our final  output per head. And if we print the shape of  
43:2043 minutes, 20 secondsthe output from each head, we see torch size of  112464. Perfect. Each of our 12 specialists has  
43:2843 minutes, 28 secondsdone its job. Step three, merging the heads.  Time to bring the team back together. First,  
43:3643 minutes, 36 secondswe transpose and reshape to merge the heads.  The code is merged_output equals output_per  
43:4443 minutes, 44 secondshead.transpose12 followed by contiguous and  viewbtc. And if we check the shape of the  
43:5243 minutes, 52 secondsmerged output, it's back to 14768. Then we pass it  through the final projection layer. C_proj equals  
44:0344 minutes, 3 secondsNN.linear CC and final_output equals C_proj of  our merged output. The shape of the final output  
44:1344 minutes, 13 secondsis 14768. Here's the entire journey in one table.  First, we split the heads going from BTC to BNHTD  
44:2544 minutes, 25 secondsto prepare for parallel computation. Second, we  run a tension where each head computes context  
44:3144 minutes, 31 secondsindependently keeping the same shape. Third, we  merge the heads going back from BNHTD to BTC to  
44:3944 minutes, 39 secondscombine all the insights. And finally, a final  projection mixes the information keeping the BTC  
44:4644 minutes, 46 secondsshape. And we're back. Each token's vector now  contains the combined contextaware information  
44:5544 minutes, 55 secondsfrom all 12 attention heads. [Music] Part two  encapsulating in the NN do module. Let's look at  
45:0545 minutes, 5 secondsthe init method. First we add the final projection  layer and an assert to make sure the dimensions  
45:1045 minutes, 10 secondsare compatible. Inside our causal self attention  class, we have self.cpoints equals nn.linear from  
45:1945 minutes, 19 secondsconfig.n n prime to config.n calls it. Simple as  that. Now for the forward method. Part one is the  
45:2845 minutes, 28 secondssplit. We get our B, T, and C from the input  size. We get Q, K, and V. And then we do the  
45:3545 minutes, 35 secondsexact same view and transpose operations we just  walked through manually. Q equals Q do.bt self.n  
45:4445 minutes, 44 secondsheaddarms and then.transpose12. We do the same  for K and V. Next, the forward method part two,  
45:5345 minutes, 53 secondsattend. This should look very familiar.  We run causal self attention on each head  
46:0046 minutesin parallel. We calculate at with the matrix  multiplication and scaling. We apply our causal  
46:0746 minutes, 7 secondsmask using masked fill. We run softax and we get  our output y by multiplying at and v. And finally,  
46:1646 minutes, 16 secondsthe forward method, part three, merge. We take  our output Y, we transpose it, make it contiguous,  
46:2346 minutes, 23 secondsand view it back into the BTC shape. Then we pass  it through our final projection layer. Y equals  
46:3046 minutes, 30 secondsself. CROJ of Y, and we return Y. Every single  line in this module should now be crystal clear.  
46:4046 minutes, 40 secondsWe have successfully built the communication  layer of the transformer. That was the causal  
46:4646 minutes, 46 secondsself-attention module. It's the big group meeting  where all the tokens get together and share  
46:5246 minutes, 52 secondsinformation. But after a big meeting, what do you  do? You go back to your desk to sit and think. You  
Chapter 9: The MLP Layer
47:0247 minutes, 2 secondsprocess all that new information on your own. This  is the exact role of our next component, the MLP,  
47:1147 minutes, 11 secondsthe thinking layer. And after the sheer complexity  of attention, this is refreshingly simple. Check  
47:2047 minutes, 20 secondsout the code we're about to build from gpt2in. py.  We have a class mlp that inherits from nn.odule.  
47:3147 minutes, 31 secondsInside the init we have a fully connected layer  called FC, a projection layer called proje and a  
47:3947 minutes, 39 secondsdropout layer. And the forward pass is just a few  lines. That's it. So let's break down the roles.  
47:4847 minutes, 48 secondsThe attention layer allows tokens to interact  with each other. That's the communication part.  
47:5547 minutes, 55 secondsThe MLP layer however processes information for  each token independently. This is the computation  
48:0348 minutes, 3 secondsor thinking part. This structure communication  followed by individual computation repeated  
48:1148 minutes, 11 secondsover and over is what gives the transformer  its power. The standard architecture for the  
48:1848 minutes, 18 secondsMLP in a transformer follows a pattern I call  expand and contract. Let's walk through it. One,  
48:2648 minutes, 26 secondsexpansion. Using the FC layer, we project from our  embedding dimension n embed dimension up to four  
48:3548 minutes, 35 secondstimes that size. Two, nonlinearity. We apply an  activation function, in this case, jello. Three,  
48:4548 minutes, 45 secondscontraction. Using the prod layer, we project that  huge vector back down to the original n embedd  
48:5448 minutes, 54 secondssize and four dropout for regularization. Okay,  before we build the whole MLP, let's demystify  
49:0349 minutes, 3 secondsits core component n.linear. It's honestly just  simple math. The output is just the input times  
49:1349 minutes, 13 secondsthe weight matrix transposed plus a bias. No more  magic. Let's prove it. Our problem. We want to  
49:2349 minutes, 23 secondsdemystify nn.linear by projecting a vector of  size 2 up to a vector of size 4. We'll import  
49:3049 minutes, 30 secondstorch and torch.n. We set our input channels C  in to two and our output channels C out to four.  
49:3949 minutes, 39 secondsThen we create our linear layer. Now this layer  has two learnable parameters. First the weights  
49:4549 minutes, 45 secondswhich is a matrix of shape C out by C in. So 4x  two for us. Second the biases which is a vector  
49:5349 minutes, 53 secondsof shape C out. So just four. To make this super  clear, let's manually set the weights and biases  
49:5949 minutes, 59 secondsto some nice simple numbers. For the weights,  we'll set the data to a torch tensor. The first  
50:0550 minutes, 5 secondsrow is 1 0. The second is - 1 0. The third is  02. And the fourth is 0 -2. For the biases,  
50:1650 minutes, 16 secondswe'll just use 1 1 -1 and -1. Finally,  our input vector will be 0.5 and0.5. Okay,  
50:2650 minutes, 26 secondslet's manually calculate just the first output  element together. Output at index 0 equals input  
50:3350 minutes, 33 secondsat 0 * weight at 0 comma 0 plus input at 1 *  weight at 0.1 plus the bias at 0. Plugging in  
50:4450 minutes, 44 secondsthe numbers, that's 0.5 * 1.0 + - 0.5 * 0.0  + 1.0, which simplifies to 0.5 + 0 + 1.0,  
50:5950 minutes, 59 seconds0 which equals 1.5. Now let's see what PyTorch  gives us. We call output vector equals linear  
51:0851 minutes, 8 secondslayer of our input vector and we print them both  out. The input is 0.5 minus 0.5 and the output  
51:1651 minutes, 16 secondsis 1.5 0.5 -2.0 and 0.0. It matches perfectly.  That's it. See? No magic at all. Okay, now let's  
51:3051 minutes, 30 secondstrace a single token's journey through the entire  MLP. Our input will be a single token vector with  
51:3851 minutes, 38 secondsC equals 2. The MLP will expand it to 4 * C,  which is 8. So our input X is a tensor of shape  
51:4651 minutes, 46 seconds1 2 with the values 0.5 and0.5. Step one, the  expansion layer FC. This is an NN.linear linear  
51:5751 minutes, 57 secondslayer going from two dimensions to eight. We'll  set a manual seed for reproducibility and create  
52:0252 minutes, 2 secondssome random weights and biases. Then we calculate  Xpanded equals FC of X. And after the expansion  
52:1052 minutes, 10 secondslayer, the shape is now torch size of 18. And the  values are something like 0.52 minus 1.02,0.32
52:2252 minutes, 22 secondsand so on. Our vector has successfully been  expanded. Step two, the GelU activation. Gelu  
52:3252 minutes, 32 secondsis just a smooth alternative to ReLU that squashes  negative values towards zero. If you look at the  
52:3952 minutes, 39 secondstable, an input of 2.0 becomes about 1.95. But  an input of -1.0 gets squashed down to about0.04.
52:5052 minutes, 50 secondsSo let's apply it. X_activated  equals F.GELU of X_expanded.
52:5952 minutes, 59 secondsAfter Gelu, the shape is still 118. But  look at the values. The positive numbers  
53:0653 minutes, 6 secondsare mostly the same, but the negative ones like  minus1.02 have been squashed down to minus0.04.
53:1553 minutes, 15 secondsThis is that critical nonlinear step.   Step three, the contraction layer prod. Now we  just project that eight-dimensional vector back  
53:2653 minutes, 26 secondsdown to our original two dimensions. Proj equals  n inn.linear from 8 to 2. We set another seed,  
53:3553 minutes, 35 secondscreate some random data for it, and then  x projected equals prod of x activated.  
53:4253 minutes, 42 secondsAfter the contraction layer, the shape is back to  torch. is 1 2 and the values are now 0.33 and0.15.
53:5653 minutes, 56 secondsAnd finally, step four, drop out. The last step  is to apply the dropout layer. During training,  
54:0454 minutes, 4 secondsthis layer randomly sets some elements to  zero for regularization. But during inference,  
54:1054 minutes, 10 secondswhen we're actually generating text, the  dropout layer does absolutely nothing.  
54:1654 minutes, 16 secondsit just passes the data through. So the final  result, our input was a vector with 0.5 and0.5.  
54:2554 minutes, 25 secondsIt went through the MLP's thinking process and  the output is a new vector with 0.33 and0.15.
54:3654 minutes, 36 secondsThe key takeaway here is that the MLP transforms  the input vector while preserving its shape.  
54:4354 minutes, 43 secondsIt goes from B T C and it ends up as B T C. And  this is super critical because it allows us to  
54:5554 minutes, 55 secondsadd this output back to the original input using  something called a residual connection which we'll  
55:0255 minutes, 2 secondsexplore next. So let's recap. We have now built  both major components of our transformer block.  
55:0955 minutes, 9 secondsWe have causal self attention for communication  and the MLP for thinking. Now we need to assemble  
55:1755 minutes, 17 secondsthem. We need to put them together into a robust  repeatable block. [Music] But we can't just stack  
55:2655 minutes, 26 secondsthem one after another. To make deep learning  actually work, we need a special ingredient.
Chapter 10: Residual Connections
55:3555 minutes, 35 secondsWe need the residual connection. Our entire  focus in this section is on the humble plus  
55:4355 minutes, 43 secondssign. Let's look at our gpt2_mmin. py file inside  the block class. The class has an init method with  
55:5155 minutes, 51 secondslayer norm attention another layer norm and an  mlp. But the magic is in the forward method. Look  
55:5955 minutes, 59 secondsat this line right here. This is our focus.  The addition operation x= x + self.add add  
56:0856 minutes, 8 secondsattend open parenthesis self.l1 open parenthesis  x close parenthesis close parenthesis and look it  
56:1756 minutes, 17 secondshappens again right after x= x plus self.mlp open  parenthesis self.ln2 ln2 close parenthesis open  
56:2556 minutes, 25 secondsparenthesis x close parenthesis. This simple  pattern x= x + something is one of the most  
56:3556 minutes, 35 secondsimportant breakthroughs in the entire history of  deep learning. So why why can't we just do this x  
56:4356 minutes, 43 secondsgoes to attention then to the MLP then to the next  block? Why do we need that extra plus x? Well,  
56:5256 minutes, 52 secondswhen you build a very deep network, you run into  a huge problem called the vanishing gradient. It's  
57:0057 minuteslike a long game of telephone. You know how that  works, right? The learning signal has to travel  
57:0557 minutes, 5 secondsbackward all the way from the final output through  every single layer back to the very first one.  
57:1257 minutes, 12 secondsWith each step backward through a layer, the  signal gets a little distorted or weaker and  
57:2057 minutes, 20 secondsweaker until it has vanished. The early layers  of the network learn absolutely nothing. The  
57:2857 minutes, 28 secondsresidual connection provides a brilliant solution  by creating a shortcut or an express lane. Look at  
57:3757 minutes, 37 secondsthis diagram. Your input X comes in at the top  left. It flows into a sub layer which contains  
57:4357 minutes, 43 secondslayer norm and causal self attention. That sub  layer does its computation and its output goes  
57:4957 minutes, 49 secondsto this plus sign. But here's the magic. Look at  this other arrow. The original input X also flows  
57:5957 minutes, 59 secondsdirectly to that plus sign completely bypassing  the sub layer. That's the residual connection,  
58:0858 minutes, 8 secondsthe express lane. During back propagation,  the gradient can flow directly through this  
58:1458 minutes, 14 secondsaddition operator. It completely bypasses all the  transformations inside the attention layer. It's  
58:2258 minutes, 22 secondsan uninterrupted highway for the gradient  to travel back in time. This fundamentally  
58:2858 minutes, 28 secondschanges what the network is trying to learn.  Without residuals, you're telling the network,  
58:3458 minutes, 34 seconds"Here is a blank canvas. Paint a masterpiece. Good  luck with that." With residuals, it's easy. You're  
58:4258 minutes, 42 secondssaying, "Here is the current painting, which  is X." Now, just make these small adjustments,  
58:4858 minutes, 48 secondswhich is the output of self.lbl. It is so  much easier for a network to learn how to  
58:5458 minutes, 54 secondsmake small iterative adjustments than it is to  learn the entire transformation from scratch   at every single layer. Let's do a walkthrough  with numbers to see just how simple this is.
59:0759 minutes, 7 secondsThe operation is just a simple element-wise  addition. We'll start with an initial input vector  
59:1359 minutes, 13 secondsfor a single token. We'll call it X initial.  Let's say it's a tensor with the values 0.2,  
59:2059 minutes, 20 seconds0.1, 0.3, and 0.4. [Music] Now that input  goes through the attention sub layer and it  
59:2859 minutes, 28 secondsproduces an output. We'll call it attention  output. This is the adjustment to be made.  
59:3559 minutes, 35 secondsLet's say its values are 0.1.2
59:4059 minutes, 40 secondsand0.3. Now for the residual  connection x= x + let's look  
59:4859 minutes, 48 secondsat the result. The original input x was 0.2 0.1.4.
59:5659 minutes, 56 secondsThe adjustment from attention  was 0.1 minus 0.1.2.3.
1:00:041 hour, 4 secondsSo the new value is just the sum. The value after  the first residual connection is a tensor with  
1:00:101 hour, 10 seconds0.3, 0.0, 0.5, and 0.1. It's that simple. It's  just addition. Critically, the shape of the tensor  
1:00:221 hour, 22 secondsremains unchanged. Look at this table. In step  one, the self.attension operation takes an input  
1:00:291 hour, 29 secondsof shape BTC and produces an output of shape BTC.  Its whole purpose is just to calculate the update  
1:00:371 hour, 37 secondsor the residual. Then in step two, the X plus  operation takes that same shape BTC and outputs,  
1:00:461 hour, 46 secondsyou guessed it, BTC. Its only job is to apply  the update to the original. We have now added  
1:00:531 hour, 53 secondsthe first piece of glue to our block. This express  lane is what allows us to build much deeper and  
1:00:591 hour, 59 secondsmuch more powerful models. But a highway with no  rules can lead to chaos. Let's dive back into our  
Chapter 11: Layer Normalization
1:01:081 hour, 1 minute, 8 secondscode. Let's look at our block class again. Our  focus this time is on these two lines self.l_1  
1:01:151 hour, 1 minute, 15 secondsand self.l_2 and more importantly where they are  applied inside the init method. You can see we  
1:01:241 hour, 1 minute, 24 secondsdefined them right here. We have self.l_1 equals  inn.layerm and self.l_2 another inn.layerm. Now  
1:01:361 hour, 1 minute, 36 secondshere's the key part. Look at the forward method.  We apply layer norm before the data goes into the  
1:01:421 hour, 1 minute, 42 secondssub layer. See it's x + self.tts of self.l_1  of x. And then we do it again before the MLP  
1:01:541 hour, 1 minute, 54 secondslayer. It's X plus self.mlp of self.l_2 of X. So  what on earth is this doing? Well, it's solving a  
1:02:051 hour, 2 minutes, 5 secondsreally big problem. The problem of trying to hit a  moving target. As data flows through the network,  
1:02:121 hour, 2 minutes, 12 secondsthe distribution at each layer is constantly  changing during training. the mean, the standard  
1:02:181 hour, 2 minutes, 18 secondsdeviation, it's all over the place. This is called  internal covariant shift. Think of it like this.  
1:02:261 hour, 2 minutes, 26 secondsEach layer in our network is trying to learn,  but the target it's aiming for is constantly  
1:02:321 hour, 2 minutes, 32 secondsmoving. It makes it incredibly difficult for the  network to learn anything efficiently. It's like  
1:02:381 hour, 2 minutes, 38 secondstrying to shoot arrows at a target strapped  to a bucking bronco. Good luck with that.  
1:02:451 hour, 2 minutes, 45 secondsLayer normalization is the technique that tames  the Bronco. It acts as a stabilizer for each  
1:02:521 hour, 2 minutes, 52 secondsindividual tokens vector. This is really  important. It performs these steps. First,  
1:02:581 hour, 2 minutes, 58 secondsit calculates the mean, which we'll call  mu, and the variance sigma squared across   that single vector's embedding  dimension. That's the C dimension.  
1:03:091 hour, 3 minutes, 9 secondsSecond, it normalizes that vector to have a mean  of zero and a standard deviation of 1. Look at  
1:03:161 hour, 3 minutes, 16 secondsthis formula. The new vector xhat equals x - mu  divided by the square root of sigma squar plus a  
1:03:251 hour, 3 minutes, 25 secondstiny little epsilon to prevent dividing by zero.  And third, it applies learnable parameters gamma  
1:03:321 hour, 3 minutes, 32 secondsand beta to scale and shift the distribution back.  The final output y is gamma x* xhat plus beta.  
1:03:421 hour, 3 minutes, 42 secondsThis gives the model back control after we've  stabilized everything. Okay, that was a lot of  
1:03:491 hour, 3 minutes, 49 secondstheory. Let's walk through this with numbers. This  is how it will click. [Music] We're going to trace  
1:03:581 hour, 3 minutes, 58 secondsthe process for a single tokens vector and we'll  say its embedding dimension C is equal to four.  
1:04:061 hour, 4 minutes, 6 secondsStep one, the input vector. Let's say we have this  tensor X token with the values 0.3, negative0.2,  
1:04:161 hour, 4 minutes, 16 seconds0.8, and 0.5. Right now, if we check its mean is  about 0.35 and its standard deviation is 0.41.
1:04:291 hour, 4 minutes, 29 secondsIt's a moving target. Okay, on to step two,  normalization. This is the Xhat calculation.  
1:04:381 hour, 4 minutes, 38 secondsWe take our input X token, we subtract the mean  and we divide by the standard deviation. And  
1:04:461 hour, 4 minutes, 46 secondswhat do we get? Look at this. The normalized  vector Xhat is now0.12, -1.33, 1.09, and 0.36.  
1:05:001 hour, 5 minutesAnd its mean zero. its standard deviation one.  Perfect. The target is no longer moving. Now for  
1:05:111 hour, 5 minutes, 11 secondsthe final step, step three, applying the learnable  parameters gamma and beta. When we create a layer  
1:05:201 hour, 5 minutes, 20 secondslayer like ln= nn.layer C, PyTorch initializes  two special parameters for us. The first is  
1:05:281 hour, 5 minutes, 28 secondslayernorm.weight, which is our gamma. it  starts as a tensor of all ones. The second is  
1:05:351 hour, 5 minutes, 35 secondslayerorm.bias our beta which starts as a tensor  of all zeros. So initially multiplying by one and  
1:05:431 hour, 5 minutes, 43 secondsadding zero doesn't change anything.  But let's pretend the model has been   training for a while and it has learned that  a different scaling is optimal for the next  
1:05:531 hour, 5 minutes, 53 secondslayer. Maybe it's learned that gamma should be  1.5 1 1 and 1 and beta should be 0.5 0 and 0.  
1:06:041 hour, 6 minutes, 4 secondsNow we apply this to our nice stable normalized  vector. Y = gamma * xhat plus beta. And here's  
1:06:141 hour, 6 minutes, 14 secondswhat we see. Our final output vector is now 0.32,  -1.33, 1.09, 09 and 0.36. It's a stable, clean,  
1:06:271 hour, 6 minutes, 27 secondsand optimally scaled input for the next part of  the network. Okay, one last crucial practical  
1:06:341 hour, 6 minutes, 34 secondsdetail. Porm versus postnorm. Look at this table.  We have two main styles. On the left is porm,  
1:06:431 hour, 6 minutes, 43 secondswhich is what we use in our GPT2 style model.  Its equation is X plus the sub layer of  
1:06:491 hour, 6 minutes, 49 secondslayer of X. The normalization happens before the  sub layer. On the right is postnorm the original  
1:06:571 hour, 6 minutes, 57 secondstransformer style. Its equation is layer norm of x  plus the sub layer of x. The normalization happens  
1:07:051 hour, 7 minutes, 5 secondsafter. Now here's the important part stability.  Porm is generally more stable. It makes it easier  
1:07:141 hour, 7 minutes, 14 secondsto train very deep networks without needing fancy  learning rate schedules. Postnorm can be harder to  
1:07:201 hour, 7 minutes, 20 secondstrain for very deep models and often requires a  learning rate warm-up. In our code, you see X= X  
1:07:281 hour, 7 minutes, 28 secondsplus self.at self.l1x. That is textbook porm. So  why does our code use prenorm? Because it's more  
1:07:401 hour, 7 minutes, 40 secondsrobust. By normalizing the input before it enters  the attention or MLP layers, we keep the values in  
1:07:481 hour, 7 minutes, 48 secondscheck and prevent the numbers from exploding.  It just makes training smoother. So with our  
1:07:551 hour, 7 minutes, 55 secondsresidual connections providing the express lane  and layer normalization acting as the stabilizer,  
1:08:031 hour, 8 minutes, 3 secondsthis is the payoff. We have meticulously built  every single component. The communication layer,  
Chapter 12: The Transformer Block
1:08:141 hour, 8 minutes, 14 secondsthe causal self attention, the thinking  layer, the MLP, the express lane,  
1:08:191 hour, 8 minutes, 19 secondsour residual connections, and the stabilizer  layer normalization. Now we click them together.
1:08:311 hour, 8 minutes, 31 secondsThis is the code for the block class. This is the  fundamental repeating Lego brick of the entire  
1:08:381 hour, 8 minutes, 38 secondsGPT2 model. That's it. I'm serious. That's really  it. And you already understand every single piece  
1:08:481 hour, 8 minutes, 48 secondsof it. Let's look at the blueprint of a block. The  dunder and knit method. The constructor is just an  
1:08:551 hour, 8 minutes, 55 secondsassembly list. It's like the parts list for a Lego  set. First, we have self.l_1 ln_1 equals nn.layer  
1:09:041 hour, 9 minutes, 4 secondsnorm. That's the stabilizer we just built. Then  self.attension equals causal self attention.  
1:09:111 hour, 9 minutes, 11 secondsThat's the communication layer you built. Then  another stabilizer self.l_2. And finally self.mlp  
1:09:201 hour, 9 minutes, 20 secondsequals mlp the thinking layer. You built this  too. There is zero new complexity here. It's  
1:09:291 hour, 9 minutes, 29 secondsjust a container for the components we've already  mastered. The forward method orchestrates the flow  
1:09:361 hour, 9 minutes, 36 secondsof data through these components and it follows  a simple powerful mantra. Normalize, process,  
1:09:441 hour, 9 minutes, 44 secondsadd. Let's trace the data's journey. Check out  this flow diagram. The input X comes in. It flows  
1:09:531 hour, 9 minutes, 53 secondsinto layer norm one. then causal self attention  and then it gets added back together with the  
1:10:011 hour, 10 minutes, 1 secondoriginal input. That output then flows into layer  norm 2 then into the MLP and gets added back again  
1:10:111 hour, 10 minutes, 11 secondsto produce the final output. Let's look at that  first sub layer the attention meeting. The code  
1:10:171 hour, 10 minutes, 17 secondsis x= x + selfide of self.l_1 of x. Let's break it  down. First, we normalize. We apply self.ln_1 to  
1:10:311 hour, 10 minutes, 31 secondsx. The input is stabilized. Simple enough. Second,  we process. That stabilized data enters the  
1:10:401 hour, 10 minutes, 40 secondsself.appro, our communication layer. And third,  we add. The result of all that is added back to  
1:10:481 hour, 10 minutes, 48 secondsthe original X. That's the residual connection  at work. Now for sub layer 2, the thinking time.  
1:10:561 hour, 10 minutes, 56 secondsThe code is x= x + self.mlp of self.ln_2 of x.  It's the exact same pattern. First normalize.  
1:11:081 hour, 11 minutes, 8 secondsWe stabilize the output from the first sub layer  using self.ln_2. Second process. Each token thinks  
1:11:181 hour, 11 minutes, 18 secondsabout the new information inside the self.mlp. And  third add the output is added back again. It's so  
1:11:271 hour, 11 minutes, 27 secondselegant. The most critical property of this entire  block is that the shape of the output tensor is  
1:11:351 hour, 11 minutes, 35 secondsidentical to the shape of the input tensor. B T  C always. This is what makes the block perfectly  
1:11:441 hour, 11 minutes, 44 secondsstackable. This stackability is the key to the  transformer's power. A single block is like a  
1:11:511 hour, 11 minutes, 51 secondssingle project meeting. It's a good first step,  but complex problems require more. To build a deep  
1:11:581 hour, 11 minutes, 58 secondsunderstanding, we need a series of meetings, a  symposium. For example, meeting one or block zero  
1:12:051 hour, 12 minutes, 5 secondsfigures out basic grammar and syntax. Then meeting  two, block one, takes that and pieces together  
1:12:131 hour, 12 minutes, 13 secondshigher level ideas. You fast forward all the way  to meeting 12, block 11, and it achieves deep,  
1:12:211 hour, 12 minutes, 21 secondsnuanced, semantic understanding. This allows us to  distinguish between width and depth. Look at this  
1:12:281 hour, 12 minutes, 28 secondstable. Let's walk through it. First, the concept  of width, which comes from multi head attention.  
1:12:351 hour, 12 minutes, 35 secondsWhat does it do? It's parallel processing  within a layer. The analogy is a committee of 12  
1:12:421 hour, 12 minutes, 42 secondsspecialists in one meeting. Why? To analyze input  from many perspectives at once. Now let's look  
1:12:521 hour, 12 minutes, 52 secondsat depth which comes from multi-layer models like  this. What does it do? It's sequential processing  
1:13:001 hour, 13 minutesacross layers. The analogy here is a series of  12 meetings each refining the last. And why do  
1:13:071 hour, 13 minutes, 7 secondswe do this? to build a hierarchical understanding  from simple patterns to abstract concepts. So how  
1:13:141 hour, 13 minutes, 14 secondsdo we code this series of meetings? We use a  special PyTorch container called nnn.module  
1:13:231 hour, 13 minutes, 23 secondslist in the main GPT2 models dunder init method.  You'll see this line self.h h equals nn.module  
1:13:321 hour, 13 minutes, 32 secondslist containing a list comprehension block  of config for underscore in range of config n  
1:13:401 hour, 13 minutes, 40 secondslayer. This simple line creates n layer say 12  separate instances of our block. This brings us  
1:13:491 hour, 13 minutes, 49 secondsto a crucial final question. Are the weights  shared between these 12 blocks? Absolutely  
1:13:571 hour, 13 minutes, 57 secondsnot. Each call to block of config creates 12  independent objects, each with its own unique  
1:14:051 hour, 14 minutes, 5 secondsweights. Why is this essential? Well, think about  it. Layer 1 is processing raw embeddings. Layer 12  
1:14:131 hour, 14 minutes, 13 secondsis making final refinements. Different layers  need different skills. By giving each block  
1:14:191 hour, 14 minutes, 19 secondsits own set of weights, we allow each layer to  specialize in its part of the job. We have now  
1:14:261 hour, 14 minutes, 26 secondsofficially built our fundamental Lego brick and  we understand the strategy for stacking them.
1:14:351 hour, 14 minutes, 35 secondsWe have officially reached the summit. After  building every single component from the  
1:14:411 hour, 14 minutes, 41 secondsground up, we are now ready to assemble the final  model. This section will feel like a victory lap.  
1:14:501 hour, 14 minutes, 50 secondsYou are going to recognize every single  line of code we look at. You built this.
1:15:041 hour, 15 minutes, 4 secondsLet's look at the init method of our final GPT2  class. This is the master blueprint, the one that  
1:15:131 hour, 15 minutes, 13 secondsorganizes every single layer we've discussed. Here  it is class gpt2 which inherits from nnn.module.  
1:15:261 hour, 15 minutes, 26 secondsInside the dunder init method we take in our  config object. We see three distinct parts.  
1:15:341 hour, 15 minutes, 34 secondsThis constructor perfectly follows the high-level  architecture we set out to build. Check out this  
1:15:411 hour, 15 minutes, 41 secondsdiagram. We start with input token ids which flow  into token and positional embeddings then through  
1:15:511 hour, 15 minutes, 51 secondsdropout into our n transformer blocks. After that  there's a final layer norm then the language model  
1:16:011 hour, 16 minutes, 1 secondhead and finally our output logits. It all fits  together. Let's walk through this init method  
1:16:101 hour, 16 minutes, 10 secondssection by section and connect every line back  to the section where you mastered it. First up,  
1:16:201 hour, 16 minutes, 20 secondspart one, the input layers. This is the entry  point of our model, converting raw token IDs  
1:16:271 hour, 16 minutes, 27 secondsinto meaningful vectors. We have self.wte, our  word vector dictionary that we built earlier.  
1:16:341 hour, 16 minutes, 34 secondsThen self.wpe, WP the positional lookup table.  You remember that from earlier too. And finally,  
1:16:421 hour, 16 minutes, 42 secondsself.drop for regularization to prevent  overfitting. Next, part two, the core processing  
1:16:491 hour, 16 minutes, 49 secondslayers. This is the heart of the model, the  deep part of deep learning. Here we have self.h.  
1:16:581 hour, 16 minutes, 58 secondsThis single powerful line creates n layer say 12  independent instances of our block class. Data  
1:17:091 hour, 17 minutes, 9 secondsflows through these blocks sequentially becoming  more refined at each step. Just like we discussed  
1:17:161 hour, 17 minutes, 16 secondsand that brings us to part three, the output  layers. These are the final preparations before  
1:17:221 hour, 17 minutes, 22 secondsmaking a prediction. We have self.l_f. This is a  final layer normalization that provides one last  
1:17:321 hour, 17 minutes, 32 secondsstabilization step. It ensures the output vectors  are well behaved before the final prediction. We  
1:17:391 hour, 17 minutes, 39 secondsmastered this concept earlier as well. We have  now defined almost all the structural components  
1:17:451 hour, 17 minutes, 45 secondsof our model. We've built the entrance,  the main processing tower, and the final  
1:17:511 hour, 17 minutes, 51 secondsstabilization stage. The only piece missing is  the most important one for a language model. The  
1:17:581 hour, 17 minutes, 58 secondslayer that actually makes the prediction. How  do we go from these complex refined vectors back  
1:18:051 hour, 18 minutes, 5 secondsto a probability distribution over our entire  vocabulary? [Music] We have reached the final  
Chapter 13: LM Head & Weight Tying
1:18:131 hour, 18 minutes, 13 secondsarchitectural component of our GPT model. Our  data has traveled a long journey. It started  
1:18:201 hour, 18 minutes, 20 secondsas embeddings, then went through that deep stack  of transformer blocks and was finally stabilized  
1:18:261 hour, 18 minutes, 26 secondsby a final layer norm. We now have highly  contextaware vectors, one for each and every  
1:18:361 hour, 18 minutes, 36 secondstoken in our sequence. The shape of our tensor is  batch by sequence length by embedding dimensions  
1:18:441 hour, 18 minutes, 44 secondsor BTC. But here's the thing. This is just an  internal representation. It's not a prediction.  
1:18:541 hour, 18 minutes, 54 secondsSo how do we use these rich 768dimensional vectors  to predict the single next word? Let's examine the  
1:19:041 hour, 19 minutes, 4 secondsfinal piece of our GPT2 models init method. We've  seen all the other parts, the token embeddings,  
1:19:121 hour, 19 minutes, 12 secondsthe positional embeddings, the dropout, the  transformer blocks, the final layer norm. And  
1:19:191 hour, 19 minutes, 19 secondsnow this is the final architectural piece. self.lm  head equals nn.linear. And then a very interesting  
1:19:301 hour, 19 minutes, 30 secondsline self.lm head.weight equals self.wte.weight.  weight. We'll come back to that second line in  
1:19:401 hour, 19 minutes, 40 secondsa minute. So, what is this LM head? It's actually  just a simple NN.linear layer. Nothing fancy. Its  
1:19:511 hour, 19 minutes, 51 secondsjob is to project from our internal vector space,  that's the C dimensions, into the vocabulary  
1:19:581 hour, 19 minutes, 58 secondsspace, which has vocabiz dimensions. But here is  the brilliant twist. This projection, this simple  
1:20:071 hour, 20 minutes, 7 secondslinear layer is applied independently and in  parallel to every single tokens vector along the  
1:20:161 hour, 20 minutes, 16 secondst dimension. So instead of making one prediction,  it makes t predictions. Let's look at this table  
1:20:261 hour, 20 minutes, 26 secondsto see what's happening. Our input to this layer  is the tensor we've been working with with a shape  
1:20:301 hour, 20 minutes, 30 secondsof BTC. The output from the LM head is a new  tensor we call logits. Its shape is B by T by  
1:20:391 hour, 20 minutes, 39 secondsvocab size. And what does it mean? It contains a  raw score, a logit for every possible next token  
1:20:481 hour, 20 minutes, 48 secondsfor each of the T input positions. This brings  us to a crucial question. Why make T predictions?  
1:20:551 hour, 20 minutes, 55 secondsIsn't that just wasteful? This is one of the most  brilliant design choices in the transformer. The  
1:21:031 hour, 21 minutes, 3 secondsanswer depends on whether you are training  the model or using it for generation. First,  
1:21:091 hour, 21 minutes, 9 secondslet's talk about efficient training. When we're  training, we want the model to predict the next  
1:21:141 hour, 21 minutes, 14 secondsword at every single position in the sequence all  at once. For example, if our input is a crane ate  
1:21:201 hour, 21 minutes, 20 secondsfish which has a t of four, the model has to learn  multiple things simultaneously. Given a, it has to  
1:21:291 hour, 21 minutes, 29 secondslearn to predict crane. Given a crane, it has to  learn to predict 8. And given a crane 8, it has  
1:21:361 hour, 21 minutes, 36 secondsto learn to predict fish. Our logits tensor gives  us all the predictions we need in a single forward  
1:21:421 hour, 21 minutes, 42 secondspass. It's incredibly clever. Look here. Logits  at index zero in the t dimension is the prediction  
1:21:501 hour, 21 minutes, 50 secondsbased on just a. We'll compare this to the target  which is crane. Then logits at index one is the  
1:21:561 hour, 21 minutes, 56 secondsprediction based on a crane. We compare that to  8. And finally logits at index 2 is the prediction  
1:22:031 hour, 22 minutes, 3 secondsbased on a crane 8 which we compare to fish. And  of course, we know this works because our causal  
1:22:091 hour, 22 minutes, 9 secondsmask ensures the prediction at any position t  only uses information from tokens zero up to t.  
1:22:161 hour, 22 minutes, 16 secondsThis parallel approach is incredibly efficient for  training on a GPU. Okay, but what about number two  
1:22:261 hour, 22 minutes, 26 secondsfor generation or inference? When we want to  generate new text, your intuition is correct.  
1:22:341 hour, 22 minutes, 34 secondswe are seemingly being wasteful. Imagine our input  is a prompt like a crane 8 where t equals 3. The  
1:22:451 hour, 22 minutes, 45 secondsmodel runs a forward pass and produces logits of  shape 1x3 by vocab size. Now here's the key part.  
1:22:531 hour, 22 minutes, 53 secondsWe throw away the predictions from the first  two positions. We only use the prediction from  
1:22:591 hour, 22 minutes, 59 secondsthe final position logits sliced at minus one to  sample the next word. This might seem inefficient,  
1:23:071 hour, 23 minutes, 7 secondsbut it's a trade-off. The architecture is heavily  optimized for the massively parallel computations  
1:23:131 hour, 23 minutes, 13 secondsneeded for training. We just leverage that same  powerful architecture for inference, even if we  
1:23:201 hour, 23 minutes, 20 secondsonly need the result from the final time step.  It's a pragmatic and very effective choice. And  
1:23:271 hour, 23 minutes, 27 secondsnow for the grand finale. Let's look very closely  at the last two lines of our models dunder init  
1:23:341 hour, 23 minutes, 34 secondsmethod again. self.lm head equals innin.linear and  then self.lm head.weight equals self.wte.weight.
1:23:491 hour, 23 minutes, 49 secondsThe second line is a simple but profound  optimization known as weight tying.  
1:23:561 hour, 23 minutes, 56 secondsIt's not just copying the weights. It is telling  PyTorch that the weight matrix for the LM head  
1:24:031 hour, 24 minutes, 3 secondsis the exact same object in memory as the  weight matrix for our token embedding layer  
1:24:101 hour, 24 minutes, 10 secondsWTE. They share the exact same weight matrix.  This means when back propagation updates one,  
1:24:181 hour, 24 minutes, 18 secondsit simultaneously updates the other. They are  tied together. So why does this make perfect  
1:24:271 hour, 24 minutes, 27 secondssense? Let's analyze the roles and shapes of these  two weight matrices. Check out this table. First,  
1:24:371 hour, 24 minutes, 37 secondsour WTE layer. Its weight attribute  self.wte.weight has a shape of vocab  
1:24:451 hour, 24 minutes, 45 secondssize by nbed. Its role is to convert a token ID,  which is a row index, into a meaning vector. It's  
1:24:531 hour, 24 minutes, 53 secondsan ID to meaning map. Now look at the lm head.  Its weight self.lm headweight also has a shape  
1:25:021 hour, 25 minutes, 2 secondsof vocab size by n embed and its role to convert  a meaning vector into a score for each token ID.  
1:25:101 hour, 25 minutes, 10 secondsIt's a meaning to ID map. They have the exact same  shape and their functions are perfectly symmetric.  
1:25:181 hour, 25 minutes, 18 secondsOne goes from ID to meaning. The other goes from  meaning back to ID. The core insight of weight  
1:25:241 hour, 25 minutes, 24 secondstying is that these two operations should be  inverses of each other. The vector that represents  
1:25:301 hour, 25 minutes, 30 secondsthe meaning of the word cat should be the same  whether cat is an input or an output. It's  
1:25:361 hour, 25 minutes, 36 secondsjust logical. And the benefits are huge. First,  massive parameter reduction for GPT2 small. This  
1:25:461 hour, 25 minutes, 46 secondsweight matrix has 50,257 rows times 768 columns.  That's roughly 38.5 million parameters. By tying  
1:25:591 hour, 25 minutes, 59 secondsthe weights, we eliminate the need to store and  train a second separate matrix of this enormous  
1:26:041 hour, 26 minutes, 4 secondssize. We just saved almost 40 million parameters  with one line of code. It's absolutely brilliant.  
1:26:141 hour, 26 minutes, 14 secondsSecond, you get improved performance. This acts  as a powerful form of regularization. By enforcing  
1:26:211 hour, 26 minutes, 21 secondsa sensible, logical architectural constraint,  it can prevent overfitting and lead to better,  
1:26:271 hour, 26 minutes, 27 secondsmore stable models. We've built the entire  GPT model architecture from the embeddings  
1:26:341 hour, 26 minutes, 34 secondsall the way to the final language model  head. We have a fully assembled machine,
1:26:421 hour, 26 minutes, 42 secondsbut right now it's an engine full of millions  of randomly initialized weights. It can't write  
1:26:481 hour, 26 minutes, 48 secondsa single coherent word. So, how do we teach it?  How does it learn? This section is all about that  
Chapter 14: Training & Loss Calculation
1:26:571 hour, 26 minutes, 57 secondsprocess. It's about the journey of data through  our model to produce a single crucial number,  
1:27:031 hour, 27 minutes, 3 secondsthe loss. This loss value quantifies exactly  how wrong the model's predictions are,  
1:27:111 hour, 27 minutes, 11 secondsand it's the signal we use to update every  single weight in the entire network. Let's  
1:27:181 hour, 27 minutes, 18 secondsfocus on the forward method. You've seen the  first half of this journey. Now, we're going  
1:27:231 hour, 27 minutes, 23 secondsto complete it. Here's the code from our GPT2  class. We've already built the init method.  
1:27:291 hour, 27 minutes, 29 secondsNow we're in the forward method which takes  self idx and an optional targets argument.  
1:27:351 hour, 27 minutes, 35 secondsWe know all the first parts the embeddings the  dropout the blocks the layer norm the lm head.  
1:27:421 hour, 27 minutes, 42 secondsThis all produces our logits tensor. Okay this is  our focus. Now you can see we initialize loss to  
1:27:501 hour, 27 minutes, 50 secondsnone. Then if targets is not none we calculate the  loss and it's all in this one line. loss equals  
1:27:591 hour, 27 minutes, 59 secondsf.cross entropy of logits doview minus1 logs do  size minus1 comma targets doview minus one. How  
1:28:111 hour, 28 minutes, 11 secondsdoes this one line teach the entire model? Let's  find out. The goal of training is beautifully  
1:28:181 hour, 28 minutes, 18 secondssimple. It's called next token prediction. Given  a sequence of words, the model's entire job is to  
1:28:261 hour, 28 minutes, 26 secondspredict the very next word. So for each chunk of  our training data, the input idx is the sequence
1:28:381 hour, 28 minutes, 38 secondsand the targets are that same sequence  just shifted one position to the left.  
1:28:451 hour, 28 minutes, 45 secondsLook at this example. If the input idx is a  crane 8, the targets are crane 8 fish. The  
1:28:551 hour, 28 minutes, 55 secondsmodel has to learn to make these predictions  at every single step in the sequence. Let's  
1:29:011 hour, 29 minutes, 1 seconddo a quick walkthrough of the forward method up  to this point. First, we get our embeddings. We  
1:29:071 hour, 29 minutes, 7 secondscreate token and positional embeddings to get our  initial BTC tensor. We've covered this x= self.we  
1:29:151 hour, 29 minutes, 15 secondsWTE of idx plus self.wp of pause. Second, we  process it through the blocks. The tensor flows  
1:29:251 hour, 29 minutes, 25 secondsthrough the stack of transformer blocks becoming  more and more contextaware for block in self.hx  
1:29:311 hour, 29 minutes, 31 secondsequals block of x. Third, we get the logits. The  final vectors are projected into our logits tensor  
1:29:391 hour, 29 minutes, 39 secondswhich has a shape of BT vocab size. Okay, now for  the main event, calculating the loss, one number  
1:29:491 hour, 29 minutes, 49 secondsto rule them all. We now have our logits which  are the predictions and our targets, the correct  
1:29:551 hour, 29 minutes, 55 secondsanswers. The final step is to compare them. Here's  that line again. Loss equals f.cross entropy. And  
1:30:031 hour, 30 minutes, 3 secondsthen we have those view calls inside. So what's  with the dot view? F.cross entropy expects 2D  
1:30:111 hour, 30 minutes, 11 secondsinputs. The dot view function is simply PyTorch's  way of squashing dimensions together to get the  
1:30:171 hour, 30 minutes, 17 secondsright shape. Let's make this crystal clear.  Look at this table. Our logits variable starts  
1:30:231 hour, 30 minutes, 23 secondswith a shape of Bt vocab size. When we call dot  view on it, we reshape it to B * T, vocab size.  
1:30:341 hour, 30 minutes, 34 secondsThe purpose of this is to create a 2D tensor where  each row is a prediction for one specific token  
1:30:411 hour, 30 minutes, 41 secondsposition. Now for targets, it starts with shape  BT. When we call dot view minus1, it becomes a 1D  
1:30:491 hour, 30 minutes, 49 secondslist of shape B * T. Its purpose is to be a simple  1D list of the correct answers, perfectly aligned  
1:30:571 hour, 30 minutes, 57 secondswith the predictions we just made. Now for the  most important part of this entire section. How  
1:31:051 hour, 31 minutes, 5 secondsdoes cross entropy actually calculate the loss?  The crucial point is that it calculates the loss  
1:31:131 hour, 31 minutes, 13 secondsfor each of the B * T predictions individually  and then [Music] it averages them to produce a  
1:31:231 hour, 31 minutes, 23 secondssingle final scalar loss value. It does not add  them up. It takes the mean. This is so important  
1:31:331 hour, 31 minutes, 33 secondsbecause it keeps the loss on a consistent scale.  Whether your batch size is 8 or 8,000, it makes  
1:31:411 hour, 31 minutes, 41 secondstraining stable. Let's put it all together.  The full training loop, a story in six stages.  
1:31:501 hour, 31 minutes, 50 secondsWe're going to tell the complete story from a raw  stream of text all the way to a weight update.  
1:31:571 hour, 31 minutes, 57 secondsLet's say our block size or t is four. Step one,  the raw data stream. Imagine our training data is  
1:32:071 hour, 32 minutes, 7 secondsjust one long river of token IDs. something like  5 12 8 21 6 33 9 4 15 7 2 and so on. Step two,  
1:32:241 hour, 32 minutes, 24 secondscreating input target pairs from that  stream. We just grab chunks. For example,  
1:32:311 hour, 32 minutes, 31 secondssample 1 might have an idx of 5 12 8 21 and  its targets would be 12 8 216. Sample 2 would  
1:32:441 hour, 32 minutes, 44 secondspick up where that one left off. Its idx would  be 6 33 94 and its targets would be 33 9415.
1:32:581 hour, 32 minutes, 58 secondsNow the batch, this is our unit of work. We stack  these samples to form a batch. Let's say with a  
1:33:061 hour, 33 minutes, 6 secondsbatch size of B= 2. This lets us process data in  parallel on a GPU. So our input idx has a shape of  
1:33:171 hour, 33 minutes, 17 seconds2x4. You can see the two samples stacked on top  of each other. And our target tensor also has a  
1:33:251 hour, 33 minutes, 25 secondsshape of 2x4 with the corresponding targets for  each sample. Okay, let's walk through this entire  
1:33:331 hour, 33 minutes, 33 secondssix-stage process together step by step. This is  how the learning actually happens. First stage  
1:33:421 hour, 33 minutes, 42 secondsone, the forward pass. The batch, our idx tensor  is fed into the model. The model processes both  
1:33:501 hour, 33 minutes, 50 secondssamples and outputs a logits tensor of shape 2x  4 by vocab size. Easy enough. Next stage two,  
1:33:591 hour, 33 minutes, 59 secondsloss reshaping. We prepare the logits and targets  for the loss function. Logits gets reshaped to  
1:34:061 hour, 34 minutes, 6 secondseight by vocab size and targets gets reshaped to  a flat list of eight numbers. 12 8 21 6 33 9 4  
1:34:171 hour, 34 minutes, 17 seconds15 Stage three cross entropy calculation. The  function calculates the loss for each of the  
1:34:251 hour, 34 minutes, 25 secondseight individual pairs. Let's imagine the results  for the first prediction. Maybe the loss is 2.5.  
1:34:321 hour, 34 minutes, 32 secondsFor the second 3.1 then 1.9 4.2 and so on. we get  eight separate loss numbers. Now stage four, the  
1:34:441 hour, 34 minutes, 44 secondsmost important one, averaging. We average these  eight individual losses. So we do 2.5 + 3.1 and so  
1:34:541 hour, 34 minutes, 54 secondson divided by 8. And let's say our final loss is  3.0. That is our single scalar loss value. Stage  
1:35:041 hour, 35 minutes, 4 secondsfive, back propagation. We call loss.backward  backward on that single scalar value 3.0. PyTorch  
1:35:131 hour, 35 minutes, 13 secondsthen calculates the gradients for every single  weight in the model based on that average loss.  
1:35:191 hour, 35 minutes, 19 secondsAnd finally, stage six, the weight update.  The optimizer takes one step. Optimizer.step  
1:35:271 hour, 35 minutes, 27 secondsupdates all the model weights, nudging them in a  direction that would have made that average loss  
1:35:321 hour, 35 minutes, 32 secondsof 3.0 a little bit lower. This entire six-stage  process is one training step. For each batch,  
1:35:401 hour, 35 minutes, 40 secondsthere is only one backward pass and one weight  update. And it's all driven by the average  
1:35:451 hour, 35 minutes, 45 secondsperformance across all the token predictions  in that batch. We have now closed the loop.  
1:35:521 hour, 35 minutes, 52 secondsWe've gone from a raw stream of text to a trained  model. We have built the entire GPT architecture.  
Chapter 15: Autoregressive Generation
1:36:021 hour, 36 minutes, 2 secondsWe have understood how it's trained. Now it's  time for the payoff. It's time to unlock its  
1:36:121 hour, 36 minutes, 12 secondstrue purpose, generating new coherent text from  scratch. The process is called auto reggressive  
1:36:211 hour, 36 minutes, 21 secondsgeneration, which sounds complex, but it's based  on a beautifully simple loop that you can probably  
1:36:271 hour, 36 minutes, 27 secondsguess right now. The core idea, the generation  loop. Let's break it down. One, predict. Feed  
1:36:381 hour, 36 minutes, 38 secondsthe current sequence in. Get a guess for the next  word. Two, sample. Roll the dice on that guess  
1:36:461 hour, 36 minutes, 46 secondsand pick one new word. Three, append. Add the new  word to the end of the sequence. And four, repeat.  
1:36:571 hour, 36 minutes, 57 secondsgo back to step one with the new longer sequence.  That's it. That's the entire algorithm. Let's do  
1:37:051 hour, 37 minutes, 5 secondsan example walkthrough. We give our trained model  the starting prompt, a crane. So iteration one,  
1:37:121 hour, 37 minutes, 12 secondsthe initial idx is 512. First predict, we feed  512 into the model. It gives us scores for the  
1:37:211 hour, 37 minutes, 21 secondsnext word. Next sampler. Let's say it gives 8  a 40% probability and lifted 35%. We roll the  
1:37:311 hour, 37 minutes, 31 secondsdice and get 8 which has an ID of 8. Finally  append. Our new sequence is now 5128. Okay.  
1:37:411 hour, 37 minutes, 41 secondsOn to iteration two. Our current sequence is a  crane 8. The current idx is 5128. So first we  
1:37:511 hour, 37 minutes, 51 secondspredict. We feed this new longer sequence 5128  into the model. Now we sample. Having seen 8,  
1:38:011 hour, 38 minutes, 1 secondthe model gives a massive 90% probability to  fish. We sample and get fish which has an ID of  
1:38:091 hour, 38 minutes, 9 seconds21. And finally we append. Our sequence becomes  528 21. This loop continues one word at a time.  
1:38:231 hour, 38 minutes, 23 secondsNow, let's connect this simple loop to the real  code from GPT2_min. py. Here's the code on screen.  
1:38:311 hour, 38 minutes, 31 secondsWe're looking at the generate method inside the  gpt2 class. Now, don't worry. We're going to  
1:38:361 hour, 38 minutes, 36 secondsbreak this whole thing down piece by piece. First,  let's look at number one at torch.nowgrad. This is  
1:38:431 hour, 38 minutes, 43 secondsa crucial optimization. It tells PyTorch not to  calculate gradients, saving massive amounts of  
1:38:501 hour, 38 minutes, 50 secondsmemory and computation. Remember, we're not  learning, we're just predicting. Number two,  
1:38:571 hour, 38 minutes, 57 secondscontext cropping. The model has a fixed context  window, the block size. If our generated sequence  
1:39:031 hour, 39 minutes, 3 secondsgets too long, we must crop it to only feed in the  most recent tokens. This is the model's short-term  
1:39:101 hour, 39 minutes, 10 secondsmemory. Number three, predict. Look at this line.  Logits equals logits colus1 colon. This is the  
1:39:201 hour, 39 minutes, 20 secondskey step. We throw away all predictions except  for the very last one. This single line connects  
1:39:281 hour, 39 minutes, 28 secondsthat wasteful training process to efficient  generation. Number four, sample f.softmax Softmax  
1:39:361 hour, 39 minutes, 36 secondsconverts our final scores into probabilities  and torch.multial performs the dice roll picking  
1:39:431 hour, 39 minutes, 43 secondsone token based on those probabilities. And  finally number five append the newly sampled  
1:39:511 hour, 39 minutes, 51 secondsnext token is concatenated to the end of our idx  sequence preparing it for the next iteration of  
1:39:571 hour, 39 minutes, 57 secondsthe loop controlling the magic the creative  knobs. This is where you become the artist.  
1:40:061 hour, 40 minutes, 6 secondsLet's look at this table. We've got two main  parameters. First, temperature. This rescales the  
1:40:111 hour, 40 minutes, 11 secondslogits before softmax. A low temperature, anything  less than 1.0 makes high probability words even  
1:40:191 hour, 40 minutes, 19 secondsmore likely. The result is more predictable  conservative text. A high temperature greater  
1:40:251 hour, 40 minutes, 25 secondsthan 1.0 flattens the distribution, making rare  words more likely. The result is more random,  
1:40:321 hour, 40 minutes, 32 secondssurprising, and creative text. Next, we have top  K. This truncates the distribution. It considers  
1:40:401 hour, 40 minutes, 40 secondsonly the K most likely tokens. So, if top K  equals 50, it throws away all but the 50 most  
1:40:471 hour, 40 minutes, 47 secondslikely words and samples from that smaller  set. The result, this prevents very rare or  
1:40:531 hour, 40 minutes, 53 secondsnonsensical words from being chosen, making  the output much more coherent. Conclusion,  
1:41:011 hour, 41 minutes, 1 secondthe transformer has clicked. We have reached  the end of our 90-minute journey. We started  
1:41:071 hour, 41 minutes, 7 secondswith this file, gpt2 min. py that seemed like a  dense, magical black box. We made a promise to  
1:41:161 hour, 41 minutes, 16 secondstake that box apart piece by piece until the magic  dissolved into understandable elegant engineering.  
1:41:241 hour, 41 minutes, 24 secondsAnd that is exactly what we did. We started with  token and positional embeddings. We built the  
1:41:301 hour, 41 minutes, 30 secondsintuition for self attention and its causal mask.  We scaled it with multi head attention. We added  
1:41:371 hour, 41 minutes, 37 secondsthe thinking MLP and glued it all together with  residual connections and layer normalization to  
1:41:441 hour, 41 minutes, 44 secondsform a block. Finally, we stacked the blocks,  added the language model head, and understood  
1:41:491 hour, 41 minutes, 49 secondstraining and generation. The GBT2_min py file  is no longer a mystery. Every at@ torch.nograd,  
1:42:001 hour, 42 minutesevery view, every plus sign now has a purpose  and a story. The transformer has clicked.

Sync to video time

All

From the series

From Zachary Huang

Learning

Related


16:21
