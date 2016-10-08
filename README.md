Ruby Natural Language Processing Resources
=======

A collection of Natural Language Processing (NLP) Ruby libraries, tools and software. Suggestions and contributions are welcome.

### Categories

* [APIs](#apis)
* [Bitext Alignment](#bitext-alignment)
* [Books](#books)
* [Case](#case)
* [Chatbot](#chatbot)
* [Classification](#classification)
* [Date and Time](#date-and-time)
* [Error Correction](#error-correction)
* [Full-Text Search](#full-text-search)
* [Keyword Ranking](#keyword-ranking)
* [Language Detection](#language-detection)
* [Machine Learning](#machine-learning)
* [Machine Translation](#machine-translation)
* [Miscellaneous](#miscellaneous)
* [Multipurpose Tools](#multipurpose-tools)
* [Named Entity Recognition](#named-entity-recognition)
* [Ngrams](#ngrams)
* [Parsers](#parsers)
* [Part-of-Speech Taggers](#part-of-speech-taggers)
* [Readability](#readability)
* [Regular Expressions](#regular-expressions)
* [Ruby NLP Presentations](#ruby-nlp-presentations)
* [Sentence Segmentation](#sentence-segmentation)
* [Speech-to-Text](#speech-to-text)
* [Stemmers](#stemmers)
* [Stop Words](#stop-words)
* [Summarization](#summarization)
* [Text Extraction](#text-extraction)
* [Text Similarity](#text-similarity)
* [Text-to-Speech](#text-to-speech)
* [Tokenizers](#tokenizers)
* [Word Count](#word-count)

## APIs

### 3rd party NLP services 
Client libraries to various 3rd party NLP API services.

* [alchemy_api](https://github.com/dbalatero/alchemy_api) - provides a client API library for AlchemyAPI's NLP services
* [aylien_textapi_ruby](https://github.com/AYLIEN/aylien_textapi_ruby) - AYLIEN's officially supported Ruby client library for accessing Text API
* [biffbot](https://github.com/tevren/biffbot) - Ruby gem for [Diffbot](http://www.diffbot.com/)'s APIs that extract Articles, Products, Images, Videos, and Discussions from any web page
* [gengo-ruby](https://github.com/gengo/gengo-ruby) - a Ruby library to interface with the Gengo API for translation
* [monkeylearn-ruby](https://github.com/monkeylearn/monkeylearn-ruby) - build and consume machine learning models for language processing from your Ruby apps
* [poliqarpr](https://github.com/apohllo/poliqarpr) - Ruby client for Poliqarp text corpus server
* [wlapi](https://github.com/arbox/wlapi) - Ruby based API for the project Wortschatz Leipzig

### Instant Messaging Bots
Client/server libraries to various 3rd party instant messengers chat bots APIs.

#### Facebook Messenger
* [facebook-messenger](https://github.com/hyperoslo/facebook-messenger) - Definitely the best Ruby client for Bots on Messenger  
* [messenger-ruby](https://github.com/netguru/messenger-ruby) - A simple library for supporting implementation of Facebook Messenger Bot in Ruby on Rails

#### Kik
* [kik](https://github.com/muaad/kik) - Build www.Kik.com bots in Ruby   

#### Microsoft Bot Framework (Skype) 
* [BotBuilder](https://dev.botframework.com/) - REST APIs (for Skype and others instant messaging apps) 
* [botframework-ruby](https://github.com/tachyons/botframework-ruby) - Microsoft Bot Framework ruby client

#### Slack

* [slack-bot-server](https://github.com/dblock/slack-bot-server) - A Grape API serving a Slack bot to multiple teams
* [slack-ruby-bot](https://github.com/dblock/slack-ruby-bot) - The easiest way to write a Slack bot in Ruby
* [slack-ruby-client](https://github.com/dblock/slack-ruby-client) - A Ruby and command-line client for the Slack Web and Real Time Messaging APIs
* [slack-ruby-gem](https://github.com/aki017/slack-ruby-gem) - A Ruby wrapper for the Slack API

#### Telegram Messenger

* [BOTServer](https://github.com/solyaris/BOTServer) - Telegram Bot API Webhooks Framework, for Rubyists
* [TelegramBot](https://github.com/eljojo/telegram_bot) - a charismatic Ruby client for Telegram's Bot API
* [TelegramBotRuby](https://github.com/shouya/telegram-bot) - yet another client for Telegram's Bot API
* [telegram-bot-ruby](https://github.com/atipugin/telegram-bot-ruby) - Ruby wrapper for Telegram's Bot API

#### Wechat
* [wechat](https://github.com/Eric-Guo/wechat) API, command and message handling for [WeChat](http://admin.wechat.com/wiki/index.php?title=Guide_for_Message_API) in Rails
* [wechat-api](https://github.com/lazing/wechat-api) - 用于微信 api 调用（非服务端推送信息）的处理。


### Dialog Systems

* [api-ai-ruby](https://github.com/api-ai/api-ai-ruby) - A Ruby SDK to the https://api.ai natural language processing service
* [expando](https://github.com/expando-lang/expando) - A translation language for defining user utterance examples in conversational interfaces (for [Api.ai](https://api.ai) and similars).
* [wit-ruby](https://github.com/modeset/wit-ruby) - Easy interface for wit.ai natural language parsing

### Voice-based devices bots
Client/server libraries to various 3rd party voice-based devices APIs.

#### Amazon Echo Alexa skills
* [alexa-home](https://github.com/zachfeldman/alexa-home) - Using Amazon Echo to control the home! 
* [Alexa-Hue](https://github.com/sarkonovich/Alexa-Hue) - Control Hue Lights with Alexa
* [alexa-rubykit](https://github.com/damianFC/alexa-rubykit) - Amazon Echo Alexa's App Kit Ruby Implementation
* [alexa-skill](https://github.com/skierkowski/alexa-skill) - A Ruby based DSL to create new Alexa Skills
* [alexa_skills_ruby](https://github.com/DanElbert/alexa_skills_ruby) - Simple library to interface with the Alexa Skills Kit

#### Google Home
Coming soon (December 2016, see: https://developers.google.com/actions/)


## Books

* [Mastering Regular Expressions](http://isbn.directory/book/9780596528126) - by Jeffrey E. F. Friedl
* [Regular Expressions Cookbook](http://isbn.directory/book/9781449319434) - by Jan Goyvaerts, Steven Levithan
* [Regular Expression Pocket Reference](http://isbn.directory/book/9780596514273) - by Tony Stubblebine
* [Text Processing with Ruby](https://pragprog.com/book/rmtpruby/text-processing-with-ruby) by Rob Miller
* [Thoughtful Machine Learning: A Test-Driven Approach](http://www.amazon.com/Thoughtful-Machine-Learning-Test-Driven-Approach/dp/1449374069/ref=sr_1_1?ie=UTF8&qid=1410923833&sr=8-1&keywords=thoughtful+machine+learning) - by Matthew Kirk
* [Understanding Computation](http://isbn.directory/book/9781449329273) - by Tom Stuart

## Bitext Alignment

Bitext alignment is the process of aligning two parallel documents on a segment by segment basis. In other words, if you have one document in English and its translation in Spanish, bitext alignment is the process of matching each segment from document A with its corresponding translation in document B.

* [alignment](https://github.com/bloomrain/alignment) - alignment functions for corpus linguistics (Gale-Church implementation)

## Case

* [active_support](https://github.com/rails/rails/tree/master/activesupport/lib/active_support) - the rails active_support gem has various string extensions that can handle case (e.g. `.mb_chars.upcase.to_s` or #transliterate)
* [string_pl](https://github.com/apohllo/string_pl) - additional support for Polish encodings in Ruby 1.9
* [twitter-cldr-rb](https://github.com/twitter/twitter-cldr-rb/blob/master/lib/twitter_cldr/shared/casefolder.rb) - casefolding
* [u](http://disu.se/software/u-1.0/) - U extends Ruby’s Unicode support
* [unicode](https://github.com/blackwinter/unicode) - Unicode normalization library
* [unicode_utils](https://github.com/lang/unicode_utils) - Unicode algorithms for Ruby 1.9

## Chatbot

* [chatterbot](https://github.com/muffinista/chatterbot) - A straightforward ruby-based Twitter Bot Framework, using OAuth to authenticate
* [Lita](https://github.com/jimmycuadra/lita) - Lita is a chat bot written in Ruby with persistent storage provided by Redis

## Classification

Classification aims to assign a document or piece of text to one or more classes or categories making it easier to manage or sort.

* [Classifier](https://github.com/cardmagic/classifier) - a general module to allow Bayesian and other types of classifications
* [classifier-reborn](https://github.com/jekyll/classifier-reborn) - (a fork of cardmagic/classifier) a general classifier module to allow Bayesian and other types of classifications
* [Latent Dirichlet Allocation](https://github.com/ealdent/lda-ruby) - used to automatically cluster documents into topics
* [liblinear-ruby-swig](https://github.com/tomz/liblinear-ruby-swig) - Ruby interface to LIBLINEAR (much more efficient than LIBSVM for text classification and other large linear classifications)
* [linnaeus](https://github.com/djcp/linnaeus) - a redis-backed Bayesian classifier
* [maxent_string_classifier](https://github.com/mccraigmccraig/maxent_string_classifier) - a JRuby maximum entropy classifier for string data, based on the OpenNLP Maxent framework
* [Naive-Bayes](https://github.com/reddavis/Naive-Bayes) - simple Naive Bayes classifier
* [nbayes](https://github.com/oasic/nbayes) - a full-featured, Ruby implementation of Naive Bayes
* [omnicat](https://github.com/mustafaturan/omnicat) - a generalized rack framework for text classifications
* [omnicat-bayes](https://github.com/mustafaturan/omnicat-bayes) - Naive Bayes text classification implementation as an OmniCat classifier strategy
* [stuff-classifier](https://github.com/alexandru/stuff-classifier) - a library for classifying text into multiple categories


## Lexical Databases and Ontologies
Lexical databases, knowledge-base common sense, multilingual lexicalized semantic networks and ontologies

### BabelNet
* [BabelNet API client](http://babelnet.org/guide) - API (with Ruby examples) for [BabelNet](http://babelnet.org/),  multilingual lexicalized semantic network and ontology 

### ConceptNet
* [ConceptNet API](https://github.com/commonsense/conceptnet5/wiki/API) - REST API for [ConceptNet](https://github.com/commonsense/conceptnet5/wiki)

### Mediawiki, Wikipedia
* [mediawiki-ruby-api](https://github.com/wikimedia/mediawiki-ruby-api) - Github mirror of "mediawiki/ruby/api" - our actual code is hosted with Gerrit (please see https://www.mediawiki.org/wiki/Developer_access for contributing
* [wikipedia-client](https://github.com/kenpratt/wikipedia-client) - Ruby client for the Wikipedia API http://github.com/kenpratt/wikipedia-client

### Wordnet
[ruby-wordnet](https://github.com/ged/ruby-wordnet) - A Ruby interface to the WordNet® Lexical Database. http://deveiate.org/projects/Ruby-WordNet
[rwordnet](https://github.com/doches/rwordnet) - A pure Ruby interface to the WordNet database http://www.texasexpat.net/


## Date and Time

* [Chronic](https://github.com/mojombo/chronic) - a pure Ruby natural language date parser
* [Chronic Between](https://github.com/jrobertson/chronic_between) - a simple Ruby natural language parser for date and time ranges
* [Chronic Duration](https://github.com/hpoydar/chronic_duration) - a simple Ruby natural language parser for elapsed time
* [Kronic](https://github.com/xaviershay/kronic) - a dirt simple library for parsing and formatting human readable dates
* [Nickel](https://github.com/iainbeeston/nickel) - extracts date, time, and message information from naturally worded text
* [Tickle](https://github.com/yb66/tickle) - a natural language parser for recurring events

## Error Correction

* [Chat Correct](https://github.com/diasks2/chat_correct) -  shows the errors and error types when a correct English sentence is diffed with an incorrect English sentence
* [gingerice](https://github.com/subosito/gingerice) - Ruby wrapper for correcting spelling and grammar mistakes based on the context of complete sentences

## Full-Text Search

* [ferret](https://github.com/jkraemer/ferret) - an information retrieval library in the same vein as Apache Lucene
* [ranguba](http://ranguba.org/) - a project to provide a full-text search system built on Groonga
* [Thinking Sphinx](https://github.com/pat/thinking-sphinx) - Sphinx plugin for ActiveRecord/Rails

## Keyword Ranking

* [graph-rank](https://github.com/louismullie/graph-rank) - Ruby implementation of the PageRank and TextRank algorithms
* [highscore](https://github.com/domnikl/highscore) - find and rank keywords in text

## Language Detection

* [Detect Language API Client](https://github.com/detectlanguage/detectlanguage-ruby) - detects language of given text and returns detected language codes and scores
* [whatlanguage](https://github.com/peterc/whatlanguage) - a language detection library for Ruby that uses bloom filters for speed

## Machine Learning

* [Decision Tree](https://github.com/igrigorik/decisiontree) - a ruby library which implements ID3 (information gain) algorithm for decision tree learning
* [rb-libsvm](https://github.com/febeling/rb-libsvm) - implementation of SVM, a machine learning and classification algorithm
* [RubyFann](https://github.com/tangledpath/ruby-fann) - a ruby gem that binds to FANN (Fast Artificial Neural Network) from within a ruby/rails environment
* [tensorflow.rb](https://github.com/somaticio/tensorflow.rb) - tensorflow for ruby

## Machine Translation

* [Google API Client](https://github.com/google/google-api-ruby-client) - Google API Ruby Client
* [microsoft_translator](https://github.com/ikayzo/microsoft_translator) - Ruby client for the microsoft translator API
* [termit](https://github.com/pawurb/termit) - Google Translate with speech synthesis in your terminal as ruby gem

## Miscellaneous

* [dialable](https://github.com/chorn/dialable) - A Ruby gem that provides parsing and output of North American Numbering Plan (NANP) phone numbers, and includes location & time zones
* [gibber](https://github.com/timonv/gibber) - Gibber replaces text with nonsensical latin with a maximum size difference of +/- 30%
* [hiatus](https://github.com/ahanba/hiatus) - a localization QA tool
* [language_filter](https://github.com/chrisvfritz/language_filter) - a Ruby gem to detect and optionally filter multiple categories of language
* [Naturally](https://github.com/dogweather/naturally) - Natural (version number) sorting with support for legal document numbering, college course codes, and Unicode
* [rwordnet](https://github.com/doches/rwordnet) - a pure Ruby interface to the WordNet lexical/semantic database
* [sort_alphabetical](https://github.com/grosser/sort_alphabetical) -  sort UTF8 Strings alphabetical via Enumerable extension
* [stringex](https://github.com/rsl/stringex) - some [hopefully] useful extensions to Ruby’s String class
* [twitter-text](https://github.com/twitter/twitter-text/tree/master/rb) - gem that provides text processing routines for Twitter Tweets
* [nameable](https://github.com/chorn/nameable) - A Ruby gem that provides parsing and output of person names, as well as Gender & Ethnicity matching

## Multipurpose Tools

The following are libraries that integrate multiple NLP tools or functionality.

* [nlp](https://github.com/knife/nlp) - NLP tools for the Polish language
* [NlpToolz](https://github.com/LeFnord/nlp_toolz) - Basic NLP tools, mostly based on OpenNLP, at this time sentence finder, tokenizer and POS tagger implemented, plus Berkeley Parser
* [Open NLP (Ruby bindings)](https://github.com/louismullie/open-nlp)
* [Stanford Core NLP (Ruby bindings)](https://github.com/louismullie/stanford-core-nlp)
* [Treat](https://github.com/louismullie/treat) - natural language processing framework for Ruby
* [twitter-cldr-rb](https://github.com/twitter/twitter-cldr-rb) - TwitterCldr uses Unicode's Common Locale Data Repository (CLDR) to format certain types of text into their localized equivalents
* [ve](https://github.com/Kimtaro/ve) - a linguistic framework that's easy to use
* [zipf](https://github.com/pks/zipf) - a collection of various NLP tools and libraries

## Named Entity Recognition

* [Confidential Info Redactor](https://github.com/diasks2/confidential_info_redactor) - a Ruby gem to semi-automatically redact confidential information from a text
* [ruby-ner](https://github.com/mblongii/ruby-ner) - named entity recognition with Stanford NER and Ruby
* [ruby-nlp](https://github.com/tiendung/ruby-nlp) - Ruby Binding for Stanford Pos-Tagger and Name Entity Recognizer

## Ngrams

* [N-Gram](https://github.com/reddavis/N-Gram) - N-Gram generator in Ruby
* [ngram](https://github.com/tkellen/ruby-ngram) - break words and phrases into ngrams
* [raingrams](https://github.com/postmodern/raingrams) - a flexible and general-purpose ngrams library written in Ruby

## Parsers

A natural language parser is a program that works out the grammatical structure of sentences, for instance, which groups of words go together (as "phrases") and which words are the subject or object of a verb.

* [linkparser](https://github.com/ged/linkparser) - a Ruby binding for the Abiword version of CMU's Link Grammar, a syntactic parser of English
* [Parslet](http://kschiess.github.io/parslet/) - A small PEG based parser library
* [rley](https://github.com/famished-tiger/Rley) - Ruby gem implementing a general context-free grammar parser based on Earley's algorithm
* [Treetop](https://github.com/cjheath/treetop) - a Ruby-based parsing DSL based on parsing expression grammars

## Part-of-Speech Taggers

* [engtagger](https://github.com/yohasebe/engtagger) - English Part-of-Speech Tagger Library; a Ruby port of Lingua::EN::Tagger
* [rbtagger](http://rbtagger.rubyforge.org/) - a simple ruby rule-based part of speech tagger
* [TreeTagger for Ruby](https://github.com/LeFnord/rstt) - Ruby based wrapper for the TreeTagger by Helmut Schmid

## Readability

* [lingua](https://github.com/dbalatero/lingua) - Lingua::EN::Readability is a Ruby module which calculates statistics on English text

## Regular Expressions

* [CommonRegexRuby](https://github.com/talyssonoc/CommonRegexRuby) - find a lot of kinds of common information in a string
* [regexp-examples](https://github.com/tom-lord/regexp-examples) - generate strings that match a given regular expression
* [verbal_expressions](https://github.com/ryan-endacott/verbal_expressions) - make difficult regular expressions easy

### Online resources
* [Explain Regular Expression](http://regexdoc.com/re/explain.pl) - breakdown and explanation of each part of your regular expression
* [Rubular](http://rubular.com/) - a Ruby regular expression editor

## Ruby NLP Presentations

* *Quickly Create a Telegram Bot in Ruby* [[tutorial](http://www.sitepoint.com/quickly-create-a-telegram-bot-in-ruby/)] - Ardian Haxha (2016)
* *N-gram Analysis for Fun and Profit* [[tutorial](http://www.blackbytes.info/2015/09/ngram-analysis-ruby/)] - [Jesus Castello](https://github.com/matugm) (2015)
* *Machine Learning made simple with Ruby* [[tutorial](http://www.leanpanda.com/blog/2015/08/24/machine-learning-automatic-classification/)] - [Lorenzo Masini](https://github.com/rugginoso) (2015)
* *Using Ruby Machine Learning to Find Paris Hilton Quotes* [[tutorial](http://datamelon.io/blog/2015/using-ruby-machine-learning-id-paris-hilton-quotes.html)] - [Rick Carlino](https://github.com/RickCarlino) (2015)
* *Exploring Natural Language Processing in Ruby* [[slides](http://www.slideshare.net/diasks2/exploring-natural-language-processing-in-ruby)] - [Kevin Dias](https://github.com/diasks2) (2015)
* *Natural Language Parsing with Ruby* [[tutorial](http://blog.glaucocustodio.com/2014/11/10/natural-language-parsing-with-ruby/)] - [Glauco Custódio](https://github.com/glaucocustodio) (2014)
* *Demystifying Data Science (Analyzing Conference Talks with Rails and Ngrams)* [[video RailsConf 2014](https://www.youtube.com/watch?v=2ZDCxwB29Bg) | [Repo from the Video](https://github.com/Genius/abstractogram)] - [Todd Schneider](https://github.com/toddwschneider) (2014)
* *Natural Language Processing with Ruby* [[video ArrrrCamp 2014](https://www.youtube.com/watch?v=5u86qVh8r0M) | [video Ruby Conf India](https://www.youtube.com/watch?v=oFmy_QBQ5DU)] - [Konstantin Tennhard](https://github.com/t6d) (2014)
* *How to parse 'go' - Natural Language Processing in Ruby* [[slides](http://www.slideshare.net/TomCartwright/natual-language-processing-in-ruby)] - [Tom Cartwright](https://github.com/tomcartwrightuk) (2013)
* *Natural Language Processing in Ruby* [[slides](https://speakerdeck.com/brandonblack/natural-language-processing-in-ruby) | [video](http://confreaks.tv/videos/railsconf2013-natural-language-processing-with-ruby)] - [Brandon Black](https://github.com/brandonblack) (2013)
* *Natural Language Processing with Ruby: n-grams* [[tutorial](http://www.sitepoint.com/natural-language-processing-ruby-n-grams/)] - [Nathan Kleyn](https://github.com/nathankleyn) (2013)
* *A Tour Through Random Ruby* [[tutorial](http://www.sitepoint.com/tour-random-ruby/)] - Robert Qualls (2013)

## Sentence Segmentation

Sentence segmentation (aka sentence boundary disambiguation, sentence boundary detection) is the problem in natural language processing of deciding where sentences begin and end. Sentence segmentation is the foundation of many common NLP tasks (machine translation, bitext alignment, summarization, etc.).

* [Pragmatic Segmenter](https://github.com/diasks2/pragmatic_segmenter)
* [Punkt Segmenter](https://github.com/lfcipriani/punkt-segmenter)
* [TactfulTokenizer](https://github.com/zencephalon/Tactful_Tokenizer)
* [Scapel](https://github.com/louismullie/scalpel)
* [SRX English](https://github.com/apohllo/srx-english)

## Speech-to-Text

* [att_speech](https://github.com/adhearsion/att_speech) - A Ruby library for consuming the AT&T Speech API for speech to text
* [pocketsphinx-ruby](https://github.com/watsonbox/pocketsphinx-ruby) - Ruby speech recognition with Pocketsphinx
* [Speech2Text](https://github.com/taf2/speech2text) - using Google Speech to Text API Provide a Simple Interface to Convert Audio Files

## Stemmers

Stemming is the term used in linguistic morphology and information retrieval to describe the process for reducing inflected (or sometimes derived) words to their word stem, base or root form.

* [Greek stemmer](https://github.com/skroutz/greek_stemmer) - a Greek stemmer
* [Ruby-Stemmer](https://github.com/aurelian/ruby-stemmer) - Ruby-Stemmer exposes the SnowBall API to Ruby
* [Turkish stemmer](https://github.com/skroutz/turkish_stemmer) - a Turkish stemmer
* [uea-stemmer](https://github.com/ealdent/uea-stemmer) - a conservative stemmer for search and indexing

## Stop Words

* [clarifier](https://github.com/meducation/clarifier)
* [stopwords](https://github.com/brez/stopwords) - really just a list of stopwords with some helpers
* [Stopwords Filter](https://github.com/brenes/stopwords-filter) - a very simple and naive implementation of a stopwords filter that remove a list of banned words (stopwords) from a sentence

## Summarization

Automatic summarization is the process of reducing a text document with a computer program in order to create a summary that retains the most important points of the original document.

* [Epitome](https://github.com/McFreely/epitome) - A small gem to make your text shorter; an implementation of the Lexrank algorithm
* [ots](https://github.com/deepfryed/ots) - Ruby bindings to open text summarizer
* [summarize](https://github.com/ssoper/summarize) - Ruby C wrapper for Open Text Summarizer

## Text Extraction

* [docsplit](http://documentcloud.github.io/docsplit/) - Docsplit is a command-line utility and Ruby library for splitting apart documents into their component parts
* [rtesseract](https://github.com/dannnylo/rtesseract) - Ruby library for working with the Tesseract OCR
* [Ruby Readability](https://github.com/cantino/ruby-readability) - a tool for extracting the primary readable content of a webpage
* [ruby-tesseract](https://github.com/meh/ruby-tesseract-ocr) - This wrapper binds the TessBaseAPI object through ffi-inline (which means it will work on JRuby too) and then proceeds to wrap said API in a more ruby-esque Engine class
* [Yomu](https://github.com/Erol/yomu) - a library for extracting text and metadata from files and documents using the Apache Tika content analysis toolkit

## Text Similarity

* [amatch](https://github.com/flori/amatch) - collection of five type of distances between strings (including Levenshtein, Sellers, Jaro-Winkler, 'pair distance'. Last one seems to work well to find similarity in long phrases)
* [damerau-levenshtein](https://github.com/GlobalNamesArchitecture/damerau-levenshtein) - calculates edit distance using the Damerau-Levenshtein algorithm
* [FuzzyMatch](https://github.com/seamusabshere/fuzzy_match) - find a needle in a haystack based on string similarity and regular expression rules
* [fuzzy-string-match](https://github.com/kiyoka/fuzzy-string-match) - fuzzy string matching library for ruby
* [FuzzyTools](https://github.com/brianhempel/fuzzy_tools) - In-memory TF-IDF fuzzy document finding with a fancy default tokenizer tuned on diverse record linkage datasets for easy out-of-the-box use
* [Going the Distance](https://github.com/schneems/going_the_distance) - contains scripts that do various distance calculations
* [hotwater](https://github.com/colinsurprenant/hotwater) - Fast Ruby FFI string edit distance algorithms
* [levenshtein-ffi](https://github.com/dbalatero/levenshtein-ffi) - fast string edit distance computation, using the Damerau-Levenshtein algorithm
* [TF-IDF](https://github.com/reddavis/TF-IDF) - Term Frequency - Inverse Document Frequency in Ruby
* [tf-idf-similarity](https://github.com/jpmckinney/tf-idf-similarity) - calculate the similarity between texts using tf*idf

## Text-to-Speech

* [espeak-ruby](https://github.com/dejan/espeak-ruby) - small Ruby API for utilizing 'espeak' and 'lame' to create text-to-speech mp3 files
* [Isabella](https://github.com/chrisvfritz/isabella) - a voice-computing assistant built in Ruby
* [tts](https://github.com/c2h2/tts) - a ruby gem for converting text-to-speech using the Google translate service

## Tokenizers

* [Jieba](https://github.com/mimosa/jieba-jruby) - Chinese tokenizer and segmenter (jRuby)
* [MeCab](https://github.com/markburns/mecab) - Japanese morphological analyzer [[MeCab Heroku buildpack](https://github.com/diasks2/heroku-buildpack-mecab)]
* [NLP Pure](https://github.com/parhamr/nlp-pure) - natural language processing algorithms implemented in pure Ruby with minimal dependencies
* [Pragmatic Tokenizer](https://github.com/diasks2/pragmatic_tokenizer) - a multilingual tokenizer to split a string into tokens
* [rseg](https://github.com/yzhang/rseg) - a Chinese Word Segmentation (中文分词) routine in pure Ruby
* [Textoken](https://github.com/manorie/textoken) - Simple and customizable text tokenization gem
* [thailang4r](https://github.com/veer66/thailang4r) - Thai tokenizer
* [tiny_segmenter](https://github.com/6/tiny_segmenter) - Ruby port of TinySegmenter.js for tokenizing Japanese text
* [tokenizer](https://github.com/arbox/tokenizer) - a simple multilingual tokenizer

## Word Count

* [wc](https://github.com/thesp0nge/wc) - a rubygem to count word occurrences in a given text
* [word_count](https://github.com/AtelierConvivialite/word_count) - a word counter for String and Hash in Ruby
* [Word Count Analyzer](https://github.com/diasks2/word_count_analyzer) - analyzes a string for potential areas of the text that might cause word count discrepancies depending on the tool used
* [WordsCounted](https://github.com/abitdodgy/words_counted) - a highly customisable Ruby text analyser
