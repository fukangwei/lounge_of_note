---
title: Qt之QWizardPage
categories: Qt语法详解
date: 2019-01-26 16:28:02
---
&emsp;&emsp;The `QWizardPage` class is the base class for wizard pages.<!--more-->

Header        | Since    | Inherits
--------------|----------|---------
`QWizardPage` | `Qt 4.3` | `QWidget`

### Public Functions

Return         | Function
---------------|----------
               | `QWizardPage(QWidget * parent = 0)`
`QString`      | `buttonText(QWizard::WizardButton which) const`
`virtual void` | `cleanupPage()`
`virtual void` | `initializePage()`
`bool`         | `isCommitPage() const`
`virtual bool` | `isComplete() const`
`bool`         | `isFinalPage() const`
`virtual int`  | `nextId() const`
`QPixmap`      | `pixmap(QWizard::WizardPixmap which) const`
`void`         | `setButtonText(QWizard::WizardButton which, const QString & text)`
`void`         | `setCommitPage(bool commitPage)`
`void`         | `setFinalPage(bool finalPage)`
`void`         | `setPixmap(QWizard::WizardPixmap which, const QPixmap & pixmap)`
`void`         | `setSubTitle(const QString & subTitle)`
`void`         | `setTitle(const QString & title)`
`QString`      | `subTitle() const`
`QString`      | `title() const`
`virtual bool` | `validatePage()`

### Signals

- `void completeChanged()`

### Protected Functions

Retunrn     | Function
------------|---------
`QVariant`  | `field(const QString & name) const`
`void`      | `registerField(const QString & name, QWidget * widget, const char * property = 0, const char * changedSignal = 0)`
`void`      | `setField(const QString & name, const QVariant & value)`
`QWizard *` | `wizard() const`

### Detailed Description

&emsp;&emsp;The `QWizardPage` class is the base class for wizard pages.
&emsp;&emsp;`QWizard` represents a wizard. Each page is a `QWizardPage`. When you create your own wizards, you can use `QWizardPage` directly, or you can subclass it for more control.
&emsp;&emsp;A page has the following attributes, which are rendered by `QWizard`: a title, a subTitle, and a set of pixmaps. Once a page is added to the wizard (using `QWizard::addPage()` or `QWizard::setPage()`), `wizard()` returns a pointer to the associated `QWizard` object.
&emsp;&emsp;Page provides five virtual functions that can be reimplemented to provide custom behavior:

- `initializePage()` is called to initialize the page's contents when the user clicks the wizard's `Next` button. If you want to derive the page's default from what the user entered on previous pages, this is the function to reimplement.
- `cleanupPage()` is called to reset the page's contents when the user clicks the wizard's `Back` button.
- `validatePage()` validates the page when the user clicks `Next` or `Finish`. It is often used to show an error message if the user has entered incomplete or invalid information.
- `nextId()` returns the `ID` of the next page. It is useful when creating `non-linear` wizards, which allow different traversal paths based on the information provided by the user.
- `isComplete()` is called to determine whether the `Next` and/or `Finish` button should be enabled or disabled. If you reimplement `isComplete()`, also make sure that `completeChanged()` is emitted whenever the complete state changes.

&emsp;&emsp;Normally, the `Next` button and the `Finish` button of a wizard are mutually exclusive. If `isFinalPage()` returns `true`, `Finish` is available; otherwise, `Next` is available. By default, `isFinalPage()` is `true` only when `nextId()` returns `-1`. If you want to show `Next` and `Final` simultaneously for a page (letting the user perform an `early finish`), call `setFinalPage(true)` on that page. For wizards that support early finishes, you might also want to set the `HaveNextButtonOnLastPage` and `HaveFinishButtonOnEarlyPages` options on the wizard.
&emsp;&emsp;In many wizards, the contents of a page may affect the default values of the fields of a later page. To make it easy to communicate between pages, `QWizard` supports a `field` mechanism that allows you to register a field (e.g., a `QLineEdit`) on a page and to access its value from any page. Fields are global to the entire wizard and make it easy for any single page to access information stored by another page, without having to put all the logic in `QWizard` or having the pages know explicitly about each other. Fields are registered using `registerField()` and can be accessed at any time using `field()` and `setField()`.

### Property Documentation

- `subTitle`: This property holds the `subtitle` of the page. The `subtitle` is shown by the `QWizard`, between the title and the actual page. Subtitles are optional. In `ClassicStyle` and `ModernStyle`, using subtitles is necessary to make the header appear. In `MacStyle`, the `subtitle` is shown as a text label just above the actual page. The `subtitle` may be `plain text` or `HTML`, depending on the value of the `QWizard::subTitleFormat` property. By default, this property contains an empty string. Access functions:

Return    | Function
----------|---------
`QString` | `subTitle() const`
`void`    | `setSubTitle(const QString & subTitle)`

- `title`: This property holds the `title` of the page. The `title` is shown by the `QWizard`, above the actual page. All pages should have a `title`. The `title` may be `plain text` or `HTML`, depending on the value of the `QWizard::titleFormat` property. By default, this property contains an empty string. Access functions:

Return    | Function
----------|---------
`QString` | `title() const`
`void`    | `setTitle(const QString & title)`

### Member Function Documentation

- `QWizardPage::QWizardPage(QWidget * parent = 0)`: Constructs a wizard page with the given `parent`. When the page is inserted into a wizard using `QWizard::addPage()` or `QWizard::setPage()`, the `parent` is automatically set to be the wizard.
- `QString QWizardPage::buttonText(QWizard::WizardButton which) const`: Returns the text on button `which` on this page. If a text has ben set using `setButtonText()`, this text is returned. Otherwise, if a text has been set using `QWizard::setButtonText()`, this text is returned. By default, the text on buttons depends on the `QWizard::wizardStyle`. For example, on `Mac OS X`, the `Next` button is called Continue.
- `void QWizardPage::cleanupPage() [virtual]`: This virtual function is called by `QWizard::cleanupPage()` when the user leaves the page by clicking Back (unless the `QWizard::IndependentPages` option is set). The default implementation resets the page's fields to their original values (the values they had before `initializePage()` was called).
- `void QWizardPage::completeChanged() [signal]`: This `signal` is emitted whenever the complete state of the page (i.e., the value of `isComplete()`) changes. If you reimplement `isComplete()`, make sure to emit `completeChanged()` whenever the value of `isComplete()` changes, to ensure that `QWizard` updates the enabled or disabled state of its buttons.
- `QVariant QWizardPage::field(const QString & name) const [protected]`: Returns the value of the field called `name`. This function can be used to access fields on any page of the wizard. It is equivalent to calling `wizard()->field(name)`.

``` cpp
void OutputFilesPage::initializePage() {
    QString className = field ( "className" ).toString();
    headerLineEdit->setText ( className.toLower() + ".h" );
    implementationLineEdit->setText ( className.toLower() + ".cpp" );
    outputDirLineEdit->setText ( QDir::convertSeparators ( QDir::tempPath() ) );
}
```

- `void QWizardPage::initializePage() [virtual]`: This virtual function is called by `QWizard::initializePage()` to prepare the page just before it is shown either as a result of `QWizard::restart()` being called, or as a result of the user clicking `Next`. (However, if the `QWizard::IndependentPages` option is set, this function is only called the first time the page is shown.) By reimplementing this function, you can ensure that the page's fields are properly initialized based on fields from previous pages.

``` cpp
void OutputFilesPage::initializePage() {
    QString className = field ( "className" ).toString();
    headerLineEdit->setText ( className.toLower() + ".h" );
    implementationLineEdit->setText ( className.toLower() + ".cpp" );
    outputDirLineEdit->setText ( QDir::convertSeparators ( QDir::tempPath() ) );
}
```

The default implementation does nothing.

- `bool QWizardPage::isCommitPage() const`: Returns `true` if this page is a commit page; otherwise returns `false`.
- `bool QWizardPage::isComplete() const [virtual]`: This virtual function is called by `QWizard` to determine whether the `Next` or `Finish` button should be enabled or disabled. The default implementation returns `true` if all mandatory fields are filled; otherwise, it returns `false`. If you reimplement this function, make sure to emit `completeChanged()`, from the rest of your implementation, whenever the value of `isComplete()` changes. This ensures that `QWizard` updates the enabled or disabled state of its buttons.
- `bool QWizardPage::isFinalPage() const`: This function is called by `QWizard` to determine whether the `Finish` button should be shown for this page or not. By default, it returns `true` if there is no next page (i.e., `nextId()` returns `-1`); otherwise, it returns `false`. By explicitly calling `setFinalPage(true)`, you can let the user perform an `early finish`.
- `int QWizardPage::nextId() const [virtual]`: This virtual function is called by `QWizard::nextId()` to find out which page to show when the user clicks the `Next` button. The return value is the `ID` of the next page, or `-1` if no page follows. By default, this function returns the lowest `ID` greater than the `ID` of the current page, or `-1` if there is no such `ID`. By reimplementing this function, you can specify a dynamic page order.

``` cpp
int IntroPage::nextId() const {
    if ( evaluateRadioButton->isChecked() ) {
        return LicenseWizard::Page_Evaluate;
    } else {
        return LicenseWizard::Page_Register;
    }
}
```

- `QPixmap QWizardPage::pixmap(QWizard::WizardPixmap which) const`: Returns the pixmap set for role `which`. Pixmaps can also be set for the entire wizard using `QWizard::setPixmap()`, in which case they apply for all pages that don't specify a pixmap.
- `void QWizardPage::registerField(const QString & name, QWidget * widget, const char * property = 0, const char * changedSignal = 0) [protected]`: Creates a field called `name` associated with the given `property` of the given `widget`. From then on, that `property` becomes accessible using `field()` and `setField()`. Fields are global to the entire wizard and make it easy for any single page to access information stored by another page, without having to put all the logic in `QWizard` or having the pages know explicitly about each other. If `name` ends with an asterisk (`*`), the field is a mandatory field. When a page has mandatory fields, the `Next` and/or `Finish` buttons are enabled only when all mandatory fields are filled. This requires a `changedSignal` to be specified, to tell `QWizard` to recheck the value stored by the mandatory field. `QWizard` knows the most common `Qt` widgets. For these (or their subclasses), you don't need to specify a `property` or a `changedSignal`. The table below lists these widgets:

Widget            | Property             | Change Notification Signal
------------------|----------------------|---------------------------
`QAbstractButton` | `bool checked`       | `toggled()`
`QAbstractSlider` | `int value`          | `valueChanged()`
`QComboBox`       | `int currentIndex`   | `currentIndexChanged()`
`QDateTimeEdit`   | `QDateTime dateTime` | `dateTimeChanged()`
`QLineEdit`       | `QString text`       | `textChanged()`
`QListWidget`     | `int currentRow`     | `currentRowChanged()`
`QSpinBox`        | `int value`          | `valueChanged()`

&emsp;&emsp;You can use `QWizard::setDefaultProperty()` to add entries to this table or to override existing entries. To consider a field `filled`, `QWizard` simply checks that their current value doesn't equal their original value (the value they had before `initializePage()` was called). For `QLineEdit`, it also checks that `hasAcceptableInput()` returns `true`, to honor any validator or mask. `QWizard's` mandatory field mechanism is provided for convenience. It can be bypassed by reimplementing `QWizardPage::isComplete()`.

- `void QWizardPage::setButtonText(QWizard::WizardButton which, const QString & text)`: Sets the `text` on button which to be `text` on this page. By default, the `text` on buttons depends on the `QWizard::wizardStyle`, but may be redefined for the wizard as a whole using `QWizard::setButtonText()`.
- `void QWizardPage::setCommitPage(bool commitPage)`: Sets this page to be a commit page if `commitPage` is `true`; otherwise, sets it to be a normal page. A commit page is a page that represents an action which cannot be undone by clicking `Back` or `Cancel`. A `Commit` button replaces the `Next` button on a commit page. Clicking this button simply calls `QWizard::next()` just like clicking `Next` does. A page entered directly from a commit page has its `Back` button disabled.
- `void QWizardPage::setField(const QString & name, const QVariant & value) [protected]`: Sets the `value` of the field called name to `value`. This function can be used to set fields on any page of the wizard. It is equivalent to calling `wizard()->setField(name, value)`.
- `void QWizardPage::setFinalPage(bool finalPage)`: Explicitly sets this page to be final if `finalPage` is `true`. After calling `setFinalPage(true)`, `isFinalPage()` returns `true` and the `Finish` button is visible (and enabled if `isComplete()` returns `true`). After calling `setFinalPage(false)`, `isFinalPage()` returns `true` if `nextId()` returns `-1`; otherwise, it returns `false`.
- `void QWizardPage::setPixmap(QWizard::WizardPixmap which, const QPixmap & pixmap)`: Sets the `pixmap` for role which to `pixmap`. The pixmaps are used by `QWizard` when displaying a page. Which pixmaps are actually used depend on the wizard style. Pixmaps can also be set for the entire wizard using `QWizard::setPixmap()`, in which case they apply for all pages that don't specify a pixmap.
- `bool QWizardPage::validatePage() [virtual]`: This virtual function is called by `QWizard::validateCurrentPage()` when the user clicks `Next` or `Finish` to perform some `last-minute` validation. If it returns `true`, the next page is shown (or the wizard finishes); otherwise, the current page stays up. The default implementation returns `true`. When possible, it is usually better style to disable the `Next` or `Finish` button (by specifying mandatory fields or reimplementing `isComplete()`) than to reimplement `validatePage()`.
- `QWizard * QWizardPage::wizard() const [protected]`: Returns the wizard associated with this page, or `0` if this page hasn't been inserted into a `QWizard` yet.