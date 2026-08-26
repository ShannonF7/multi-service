--
-- PostgreSQL database dump
--
CREATE EXTENSION IF NOT EXISTS vector;


-- Dumped from database version 15.14 (Ubuntu 15.14-1.pgdg22.04+1)
-- Dumped by pg_dump version 15.14 (Ubuntu 15.14-1.pgdg22.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: attraction_images; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.attraction_images (
    id integer NOT NULL,
    attraction_id integer NOT NULL,
    file_path text NOT NULL,
    upload_by character varying(255) NOT NULL,
    embedding public.vector(128)
);


--
-- Name: attraction_images_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.attraction_images_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: attraction_images_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.attraction_images_id_seq OWNED BY public.attraction_images.id;


--
-- Name: attractions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.attractions (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    description text,
    "position" character varying(50),
    jpg_path text[],
    upload_by character varying(255) NOT NULL
);


--
-- Name: attractions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.attractions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: attractions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.attractions_id_seq OWNED BY public.attractions.id;


--
-- Name: attraction_images id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attraction_images ALTER COLUMN id SET DEFAULT nextval('public.attraction_images_id_seq'::regclass);


--
-- Name: attractions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attractions ALTER COLUMN id SET DEFAULT nextval('public.attractions_id_seq'::regclass);


--
-- Name: attraction_images attraction_images_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attraction_images
    ADD CONSTRAINT attraction_images_pkey PRIMARY KEY (id);


--
-- Name: attractions attractions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attractions
    ADD CONSTRAINT attractions_pkey PRIMARY KEY (id);


--
-- Name: attraction_images attraction_images_attraction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attraction_images
    ADD CONSTRAINT attraction_images_attraction_id_fkey FOREIGN KEY (attraction_id) REFERENCES public.attractions(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

