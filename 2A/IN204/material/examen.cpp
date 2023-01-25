#include<vector>

template<typename T = std::vector<int>>  // default template value
class view
{
public:
    using container = T;
    using containerInt = typename container::int;

private:
    container& m_container;  // Référence au container stockant les valeurs.
    int m_first_index;              // Index de la première valeur de la vue.
    int m_last_index;               // Index de la dernière valeur de la vue.

public:
    using container& = std::vector<int>&;
    explicit view(container& vector) {
        m_container = vector;
        m_first_index = 0;
        m_last_index = -1;
    };

    view(container& vector, int first_index, int last_index) {
        m_container = vector;
        m_first_index = first_index;
        m_last_index = last_index;
    };


    container& getContainer(void) const {
        return this->m_container;
    };

    int getFirstIndex(void) const {
        return this->m_first_index;
    };

    int getLastIndex(void) const{
        return this->m_last_index;
    };

    bool operator == (const view& rView) const
    {
        view lView = *this;
        
        container& rVector = rView.m_container;
        int rEnd = rView.m_last_index;
        int rStart = rView.m_first_index;

        container& lVector = lView.m_container;
        int lStart = lView.m_first_index;
        int lEnd = lView.m_last_index;

        if ((lEnd - lStart) != (rEnd - rStart))
            return false;

        bool equal = true;
        for(int i = lStart; i < lEnd; i++) {
            if (lVector[i] != rVector[i])
                equal = false;
        }

        return equal;
    };

    bool operator != (const view& rView) const
    {
        view lView = *this;
        
        container& rVector = rView.m_container;
        int rEnd = rView.m_last_index;
        int rStart = rView.m_first_index;

        container& lVector = lView.m_container;
        int lStart = lView.m_first_index;
        int lEnd = lView.m_last_index;

        if ((lEnd - lStart) != (rEnd - rStart))
            return false;

        bool equal = false;
        for(int i = lStart; i < lEnd; i++) {
            if (lVector[i] != rVector[i])
                equal = true;
        }

        return equal;
    };

    void empty(void) {
        int end = this->m_last_index;
        int start = this->m_first_index;

        for(int i = start; i < end; i++) {
            this->m_container[i] = 0;
        };
    };

    int size(void) {
        return this->m_last_index - this->m_first_index;
    };

    int begin(void) {
        return this->m_first_index;
    };

    int end(void) {
        return this->m_last_index;
    };

    T operator[](int index) const {
        return this->m_container[index];
    };
};
