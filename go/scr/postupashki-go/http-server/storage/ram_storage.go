package storage

import (
	"github.com/google/uuid"
)

// @description imitation of map storage
type RamStorage struct {
	data map[uuid.UUID][]string
}

// @description creates a map storage
func NewRamStorage() *RamStorage {
	return &RamStorage{
		data: make(map[uuid.UUID][]string),
	}
}

// @title POST /task
// @description Загрузка таски на обработку генерация uuid64 и выдача его пользователю.
func (rhs *RamStorage) PostTask(value string) (*uuid.UUID, error) {
	key := uuid.New()
	if _, exists := rhs.data[key]; exists {
		return nil, ErrKeyAlreadyExists
	}
	rhs.data[key] = []string{value, "in progress"}
	_ = rhs.Sleep(key) /// mock func

	return &key, nil
}

// @description mock func
func (rhs *RamStorage) Sleep(key uuid.UUID) error {
	/*
		if rhs.data[key][1] == "in progress" {
			return "in progress", nil
		}
	*/
	rhs.data[key][1] = "ready"
	rhs.data[key][2] = "hihihaha"
	return nil
}

// @title GET /status
// @description получение статуса таски по ключу
func (rhs *RamStorage) GetStatus(key uuid.UUID) (*string, error) {
	if _, exists := rhs.data[key]; !exists {
		return nil, ErrNotFound
	}
	return &rhs.data[key][1], nil
}

// @title GET /resilt
// @description получение результата
func (rhs *RamStorage) GetResult(key uuid.UUID) (*string, error) {
	if _, exists := rhs.data[key]; !exists {
		return nil, ErrNotFound
	}
	if rhs.data[key][1] == "in progress" {
		return nil, ErrTaskIsInProgress
	}
	return &rhs.data[key][2], nil
}

// @title DELETE /task
// @description удаление завершенной задачи
func (rhs *RamStorage) Delete(key uuid.UUID) error {
	if _, exists := rhs.data[key]; !exists {
		return ErrNotFound
	}
	if rhs.data[key][1] == "in progress" {
		return ErrTaskIsInProgress
	}
	delete(rhs.data, key)
	if _, exists := rhs.data[key]; exists {
		return ErrInternalError
	}
	return nil
}
