package http

import (
	"encoding/json"
	"errors"
	"fmt"
	"http-server/storage"
	"io"
	"log"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

// @description server
type Server struct {
	storage storage.Storage
}

// @description server creation
func newServer(storage storage.Storage) *Server {
	return &Server{storage: storage}
}

// @description request body
type GetHandlerRequest struct {
	Key uuid.UUID
}

// @description parsing from request
func CreateGetHandlerRequest(r *http.Request) (*GetHandlerRequest, error) {
	key := r.URL.Query().Get("key")
	if key == "" {
		return nil, errors.New("Missing key in params")
	}

	keyId, err := uuid.Parse(key)
	if err != nil {
		return nil, errors.New("Invalid key")
	}
	return &GetHandlerRequest{Key: keyId}, nil
}

type GetHandlerResponse struct {
	Value string `json:"value"`
}

// @description get status by /status/{task_id}
func (s *Server) getStatusHandler(w http.ResponseWriter, r *http.Request) {
	req, err := CreateGetHandlerRequest(r)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
	}

	value, err := s.storage.GetStatus(req.Key)
	if err != nil {
		http.Error(w, "Key not found", http.StatusNotFound)
		return
	}

	if err := json.NewEncoder(w).Encode(GetHandlerResponse{Value: *value}); err != nil {
		http.Error(w, "Internal error", http.StatusInternalServerError)
	}
}

type PostHandlerRequest struct {
	Value string `json:"value"`
}

func CreatePostHandlerRequest(r *http.Request) (*PostHandlerRequest, error) {
	var data PostHandlerRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Failed to read request body")
		return nil, err
	}
	log.Printf("Request body: %s", body)

	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	return &data, nil
}

func (s *Server) postTaskHandler(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	data, err := CreatePostHandlerRequest(r)
	if err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	value := data.Value
	if value == "" {
		http.Error(w, "Mising value", http.StatusInternalServerError)
	}

	uuid, err := s.storage.PostTask(value)
	if err != nil {
		http.Error(w, "Failed to store value", http.StatusInternalServerError)
		return
	}

	_, _ = fmt.Fprint(w, uuid)
	w.WriteHeader(http.StatusOK)

}

func (s *Server) getResultHandler(w http.ResponseWriter, r *http.Request) {
	req, err := CreateGetHandlerRequest(r)
	if err != nil {
		http.Error(w, "Internal error", http.StatusNotFound)
	}

	value, err := s.storage.GetResult(req.Key)
	if err != nil || value == nil {
		http.Error(w, "Key not found", http.StatusNotFound)
		return
	}

	_, _ = fmt.Fprint(w, *value)
}

// @description
func CreateAndRunServer(storage storage.Storage, addr string) error {
	server := newServer(storage)

	r := chi.NewRouter()

	r.Route("/task", func(r chi.Router) {
		r.Post("/", server.postTaskHandler)
	})

	r.Route("/status", func(r chi.Router) {
		r.Get("/", server.getStatusHandler)
	})

	r.Route("/result", func(r chi.Router) {
		r.Get("/", server.getResultHandler)
	})

	httpServer := &http.Server{
		Addr:    addr,
		Handler: r,
	}

	return httpServer.ListenAndServe()
}
